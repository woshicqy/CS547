import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

# import matplotlib.pyplot as plt
import sys
import numpy as np
import random

import torchvision
import torchvision.transforms as transforms

#### for resnet18 fine tune
from torch.utils import model_zoo

from model import ResNet

from checkpoint import load_checkpoint
from tqdm import tqdm

import torch.optim as optim
from checkpoint import save_checkpoint
import torch.nn as nn

import argparse
import ast
from matplotlib import pyplot as plt

import sys

import torch.nn.parallel
import torch.distributed as dist
import torch.utils.data.distributed
from torch.multiprocessing import Pool, Process
############################################################
#                                                          # 
#  run coding as:                                          #
#  python main.py --debug False                            #
#  we hardcoded the node0-privateIP and world_size=4       #
#  but input the rank and local_rank inputs as             #
#  arg[1] and arg[2] command line arguments, respectively. #
#                                                          #
############################################################

parser = argparse.ArgumentParser()

parser.add_argument("--fine_tune", default=False, type=ast.literal_eval,
                    dest =  "fine_tune",
                    help="fine-tune pretrained model")
parser.add_argument("--lr", default=0.001, type=float, 
                    help="learning rate")
parser.add_argument("--epochs", default=40, type=int, 
                    help="number of training epochs")

parser.add_argument("--wd", default=1e-4, type=float, 
                    help="number of training epochs")

parser.add_argument("--momentum", default=0.9, type=float, 
                    help="number of training epochs")

parser.add_argument("--lr_schedule", default=True, type=ast.literal_eval, 
                    dest =  "lr_schedule",
                    help="perform lr shceduling")
parser.add_argument("--load_checkpoint", default=False, type=ast.literal_eval, 
                    dest =  "load_checkpoint",                    
                    help="resume from checkpoint")
parser.add_argument("--show_sample_image", default=False, type=ast.literal_eval, 
                    dest =  "show_sample_image",
                    help="display data insights")
parser.add_argument("--debug", default=False, type=ast.literal_eval, 
                    dest =  "debug",
                    help="using debug mode")
parser.add_argument("--data_path", default="./data", type=str, 
                    help="path to store data")
args = parser.parse_args()


def data_loader_and_transformer(root_path, fine_tune=False):
    if not fine_tune:
        train_data_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_data_tranform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    else:
        train_data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_data_tranform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    # Data loader.
    train_dataset = torchvision.datasets.CIFAR100(
        root=root_path,
        train=True,
        download=True,
        transform=train_data_transform
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        sampler=train_sampler
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root=root_path,
        train=False,
        download=True,
        transform=test_data_tranform
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )

    return train_data_loader, test_data_loader,train_sampler

############### distributed learning ##################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
#######################################################
def test(
    net, 
    criterion, 
    test_data_loader, 
    device,
    debug=False
    ):
 
    # Set to test mode.
    net.eval()

    # To monitor the testing process.
    running_loss = 0
    total_correct = 0
    total_samples = 0

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # Testing step.
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(test_data_loader):
            if debug and total_samples >= 10001:
                return
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels.long())
            prec1, prec3 = accuracy(output, target, topk=(1,3))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top3.update(prec3[0], input.size(0))
            
            # Loss.
            running_loss += loss.item()
            curt_loss = running_loss / (batch_index + 1)

            # Accuracy
            _, predict_label = torch.max(outputs, 1)
            total_samples += labels.shape[0]
            total_correct += predict_label.eq(labels.long()).float().sum().item()
            accuracy = total_correct / total_samples

            # print('Testing [batch: %d] loss: %.3f, accuracy: %.5f' %
            #         (batch_index + 1, curt_loss, accuracy))
    
    print('Testing finished accuracy: %.5f' % accuracy)
    return top1.avg



def main():
    # Set seed.
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Load data.
    print("Performing data loader...")
    train_data_loader, test_data_loader,sampler = data_loader_and_transformer(
                                                args.data_path, 
                                                fine_tune=args.fine_tune)

    # Load sample image.
    if args.show_sample_image:
        print("Loading image sample from a batch...")
        data_iter = iter(train_data_loader)
        images, labels = data_iter.next()  # Retrieve a batch of data
        
        # Some insights of the data.
        # images type torch.FloatTensor, shape torch.Size([128, 3, 32, 32])
        print("images type {}, shape {}".format(images.type(), images.shape))
        # shape of a single image torch.Size([3, 32, 32])
        print("shape of a single image", images[0].shape)
        # labels type torch.LongTensor, shape torch.Size([128])
        print("labels type {}, shape {}".format(labels.type(), labels.shape))
        # label for the first 4 images tensor([12, 51, 91, 36])
        print("label for the first 4 images", labels[:4])
        
        # Get a sampled image.
        plt.imshow(images[0][0].numpy())
        plt.savefig("sample_image.png")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'



    #### initializing ####
    print("Initialize Process Group...")
    world_size = 2
    dist_url = "tcp://172.31.22.234:23456"
    dist.init_process_group(backend=dist_backend, 
                            init_method=dist_url, 
                            rank=int(sys.argv[1]), 
                            world_size=world_size)

    local_rank = int(sys.argv[2])
    dp_device_ids = [local_rank]
    torch.cuda.set_device(local_rank)

    # Load model.
    print("*** Initializing model...")
    resnet = ResNet([2, 4, 4, 2])
    resnet = torch.nn.parallel.DistributedDataParallel(
                                      resnet, 
                                      device_ids=dp_device_ids, 
                                      output_device=local_rank)




    resnet = resnet.to(device)
    if device == 'cuda':
        resnet = torch.nn.DataParallel(resnet)
        cudnn.benchmark = True
    
    # Load checkpoint.
    start_epoch = 0
    best_acc = 0
    if args.load_checkpoint:
        print("*** Loading checkpoint...")
        start_epoch, best_acc = load_checkpoint(resnet)

    # Training.
    print('Start training on device {}...'.format(device))
    print('Hyperparameters: LR = {}, EPOCHS = {}, LR_SCHEDULE = {}'
          .format(args.lr, args.epochs, args.lr_schedule))
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(resnet.parameters(),args.lr, 
                                momentum=arg.momentum, weight_decay=arg.wd)



    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    for curt_epoch in tqdm(range(start_epoch, args.epochs)):
        sampler.set_epoch(epoch)
        train_acc = []
        test_acc = []
        resnet.train()

        ### avoiding overflow ###
        if curt_epoch > 10:
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if state['step'] >= 1024:
                        state['step'] = 1000


        running_loss = 0
        total_correct = 0
        total_samples = 0


        if args.lr_schedule:
            scheduler = optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.1)
            scheduler.step()

        
        for batch_index, (images, labels) in enumerate(train_data_loader):
        
            ### For debug ###
            if args.debug and total_samples >= 10001:
                return
            
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = resnet(images)

            
            loss = criterion(outputs, labels.long())
            prec1, prec3 = accuracy(output, target, topk=(1,3))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top3.update(prec3[0], input.size(0))



            loss.backward()

            optimizer.step()

            # Loss.
            running_loss += loss.item()
            curt_loss = running_loss / (batch_index + 1)

            # Accuracy.
            _, predict_label = torch.max(outputs, 1)
            total_samples += labels.shape[0]
            total_correct += predict_label.eq(labels.long()).float().sum().item()
            accuracy = total_correct / total_samples
            
            # print('Training [epoch: %d, batch: %d] loss: %.3f, accuracy: %.5f' %
            #         (curt_epoch + 1, batch_index + 1, curt_loss, accuracy))
            
            # Update best accuracy and save checkpoint
            if accuracy > best_acc:
                best_acc = accuracy
                save_checkpoint(resnet, curt_epoch, best_acc)
        train_acc.append(top1.avg)
        
        # Loss.
        # print('Training [epoch: %d] loss: %.3f, accuracy: %.5f' %
        #         (curt_epoch + 1, curt_loss, accuracy))

        print('Epoch: {}'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch,loss=losses, top1=top1, top5=top5))

        test_accuracy = test(resnet,criterion,
                        test_data_loader,
                        device,debug=args.debug)
        test_acc.append(test_accuracy)
        train_acc = np.array(train_acc)
        test_acc = np.array(test_acc)
        np.save('sync_sgd_train.npy',train_acc)
        np.save('sync_sgd_test.npy',test_acc)
    ################################

    # Testing.
    print("--Start testing after training...")
    final_test_acc = test(resnet,criterion,test_data_loader,device,debug=args.debug)
    print("--All work completed!--")

if __name__=="__main__":
    main()