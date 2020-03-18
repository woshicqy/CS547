import ast
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import train
# from load_data import load_data
from dataloader import load_data

from utils.utils import set_seeds, get_device, _get_device, torch_device_one
from utils import optim, configuration
import argparse


# TSA
def get_tsa_thresh(schedule, global_step, num_train_steps, start, end):
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))
    if schedule == 'linear_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)
    output = threshold * (end - start) + start
    return output.to(_get_device())


def main(args):
    # Load Configuration
    cfg = configuration.params.from_json(args.data)              # Train or Eval cfg
    model_cfg = configuration.model.from_json(args.model)        # BERT_cfg
    set_seeds(cfg.seed)

    # Load Data & Create Criterion
    data = load_data(cfg,args)
    if cfg.uda_mode:
        unsup_criterion = nn.KLDivLoss(reduction='none')
        data_iter = [data.sup_data_iter(), data.unsup_data_iter()] if cfg.mode=='train' \
            else [data.sup_data_iter(), data.unsup_data_iter(), data.eval_data_iter()]  # train_eval
    else:
        data_iter = [data.sup_data_iter()]
    sup_criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Load Model
    model = models.Classifier(model_cfg, len(data.TaskDataset.labels))
    # Create trainer
    trainer = train.Trainer(cfg, model, data_iter, optim.optim4GPU(cfg, model), get_device(),args)

    # Training
    def get_loss(model, sup_batch, unsup_batch, global_step,args):

        # logits -> prob(softmax) -> log_prob(log_softmax)

        # batch
        input_ids, segment_ids, input_mask, label_ids = sup_batch
        if unsup_batch:
            ori_input_ids, ori_segment_ids, ori_input_mask, \
            aug_input_ids, aug_segment_ids, aug_input_mask  = unsup_batch

            input_ids = torch.cat((input_ids, aug_input_ids), dim=0)
            segment_ids = torch.cat((segment_ids, aug_segment_ids), dim=0)
            input_mask = torch.cat((input_mask, aug_input_mask), dim=0)
            
        # logits
        logits = model(input_ids, segment_ids, input_mask)

        # sup loss
        sup_size = label_ids.shape[0]            
        sup_loss = sup_criterion(logits[:sup_size], label_ids)  # shape : train_batch_size
        if cfg.tsa:
            tsa_thresh = get_tsa_thresh(cfg.tsa, global_step, cfg.total_steps, start=1./logits.shape[-1], end=args.end_num)
            larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh   # prob = exp(log_prob), prob > tsa_threshold
            # larger_than_threshold = torch.sum(  F.softmax(pred[:sup_size]) * torch.eye(num_labels)[sup_label_ids]  , dim=-1) > tsa_threshold
            loss_mask = torch.ones_like(label_ids, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
            sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch_device_one())
        else:
            sup_loss = torch.mean(sup_loss)

        # unsup loss
        if unsup_batch:
            # ori
            with torch.no_grad():
                ori_logits = model(ori_input_ids, ori_segment_ids, ori_input_mask)
                ori_prob   = F.softmax(ori_logits, dim=-1)    # KLdiv target
                # ori_log_prob = F.log_softmax(ori_logits, dim=-1)

                # confidence-based masking
                if cfg.uda_confidence_thresh != -1:
                    unsup_loss_mask = torch.max(ori_prob, dim=-1)[0] > cfg.uda_confidence_thresh
                    unsup_loss_mask = unsup_loss_mask.type(torch.float32)
                else:
                    unsup_loss_mask = torch.ones(len(logits) - sup_size, dtype=torch.float32)
                unsup_loss_mask = unsup_loss_mask.to(_get_device())
                    
            # aug
            # softmax temperature controlling
            uda_softmax_temp = cfg.uda_softmax_temp if cfg.uda_softmax_temp > 0 else 1.
            aug_log_prob = F.log_softmax(logits[sup_size:] / uda_softmax_temp, dim=-1)

            # KLdiv loss
            """
                nn.KLDivLoss (kl_div)
                input : log_prob (log_softmax)
                target : prob    (softmax)
                https://pytorch.org/docs/stable/nn.html

                unsup_loss is divied by number of unsup_loss_mask
                it is different from the google UDA official
                The offical unsup_loss is diviede by total
                https://github.com/google-research/uda/blob/master/text/uda.py#L175
            """
            unsup_loss = torch.sum(unsup_criterion(aug_log_prob, ori_prob), dim=-1)
            unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1), torch_device_one())
            final_loss = sup_loss + cfg.uda_coeff*unsup_loss

            return final_loss, sup_loss, unsup_loss
        return sup_loss, None, None

    # evaluation
    def get_acc(model, batch):
        # input_ids, segment_ids, input_mask, label_id, sentence = batch
        input_ids, segment_ids, input_mask, label_id = batch
        logits = model(input_ids, segment_ids, input_mask)
        _, label_pred = logits.max(1)

        result = (label_pred == label_id).float()
        accuracy = result.mean()
        # output_dump.logs(sentence, label_pred, label_id)    # output dump

        return accuracy, result

    if cfg.mode == 'train':
        trainer.train(get_loss, None, cfg.model_file, cfg.pretrain_file)

    if cfg.mode == 'train_eval':
        trainer.train(get_loss, get_acc, cfg.model_file, cfg.pretrain_file)

    if cfg.mode == 'eval':
        results = trainer.eval(get_acc, cfg.model_file, None)
        total_accuracy = torch.cat(results).mean().item()
        print('Accuracy :' , total_accuracy)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = "UDA")
    # == hyperparameters  == #
    # Debug mode #
    parser.add_argument('--debug',               type = ast.literal_eval, default = True,     
                        dest = 'debug',
                        help = 'True or False flag for debug mode, input should be either True or False.')

    parser.add_argument('--need_prepro',         type = ast.literal_eval, default = False,     
                        dest = 'need_prepro',
                        help = 'True or False flag for preprocess, input should be either True or False.')




    ### data json file ###
    parser.add_argument('--data',                default = "config/uda.json")
    parser.add_argument('--model',               default = "config/bert_base.json")

    # optimizer #
    parser.add_argument('--learning_rate',       type = float, default = 1e-3)
    parser.add_argument('--optim_momentum_value',type = float, default = 0.9)
    parser.add_argument('--weight_decay',        type = float, default = 1e-8)


    # Training #
    parser.add_argument('--num_threads',         type = int,   default = 1) ## multi-process in cpu ##



    parser.add_argument('--epoch',               type = int,   default = 10000)
    parser.add_argument('--end_num',             type = int,   default = 1,
                    help = 'End label of this dataset, default is 1 for MDP dataset')

    parser.add_argument('--train_batch_size',    type = int,   default = 8,
                    help = 'batch_size of training')

    parser.add_argument('--eval_batch_size',     type = int,   default = 16,
                    help = 'batch_size of evaluation')



    parser.add_argument('--cuda',                type = ast.literal_eval, default = False,     
                    dest = 'cuda',
                    help = 'True or False flag, Cuda or not') ## having no cuda T_T ##


    # load epoch #
    # Saving part #
    parser.add_argument('--load_epoch',          type = str, default = 1,      metavar = "LE",
                        help='number of epoch to be loaded')

    parser.add_argument('--load_step',           type = str, default = 200,    metavar = "LS",
                        help='number of step to be loaded')


    ### Task ###
    parser.add_argument('--task',                type = str, default = 'imdb', metavar = "LS",
                        help='number of step to be loaded')

    parser.add_argument('--resume',              type = ast.literal_eval,   default = False,     
                        dest = 'resume',
                        help = "True or False flag, resume or not" )

    parser.add_argument('--weights_path',        type = str, default = "./weights/",
                        help='path to save weights')


    args = parser.parse_args()

    main(args)
