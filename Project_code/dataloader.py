import ast
import csv
import itertools

import pandas as pd    # only import when no need_to_preprocessing
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from utils import tokenization
from utils.utils import truncate_tokens_pair
class CsvDataset(Dataset):
    labels = None
    def __init__(self, file, need_prepro, pipeline, max_len, mode, d_type,cfg,args):
        Dataset.__init__(self)
        self.cnt = 0

        # need preprocessing
        if need_prepro:
            with open(file, 'r', encoding='utf-8') as f:
                lines = csv.reader(f, delimiter='\t', quotechar='"')

                # supervised dataset
                if d_type == 'sup':
                    # if mode == 'eval':
                        # sentences = []
                    data = []

                    for instance in self.get_sup(lines):
                        # if mode == 'eval':
                            # sentences.append([instance[1]])

                        if args.need_prepro:
                            tokenizer = tokenization.FullTokenizer(vocab_file=cfg.vocab, 
                                                                   do_lower_case=cfg.do_lower_case)

                            pipline = pre_pipeline(tokenizer.convert_to_unicode, 
                                                   tokenizer.tokenize,
                                                   tokenizer.convert_tokens_to_ids, 
                                                   labels,
                                                   cfg.max_seq_length)
                            instance = pipline(instance,d_type)

                            data.append(instance)


                    self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]
                    # if mode == 'eval':
                        # self.tensors.append(sentences)

                # unsupervised dataset
                elif d_type == 'unsup':
                    data = {'ori':[], 'aug':[]}
                    for ori, aug in self.get_unsup(lines):
                        for proc in pipeline:
                            ori = proc(ori, d_type)
                            aug = proc(aug, d_type)
                        self.cnt += 1
                        # if self.cnt == 10:
                            # break
                        data['ori'].append(ori)    # drop label_id
                        data['aug'].append(aug)    # drop label_id
                    ori_tensor = [torch.tensor(x, dtype=torch.long) for x in zip(*data['ori'])]
                    aug_tensor = [torch.tensor(x, dtype=torch.long) for x in zip(*data['aug'])]
                    self.tensors = ori_tensor + aug_tensor
        # already preprocessed
        else:
            f = open(file, 'r', encoding='utf-8')
            data = pd.read_csv(f, sep='\t')

            # supervised dataset
            if d_type == 'sup':
                # input_ids, segment_ids(input_type_ids), input_mask, input_label
                input_columns = ['input_ids', 'input_type_ids', 'input_mask', 'label_ids']
                self.tensors = [torch.tensor(data[c].apply(lambda x: ast.literal_eval(x)), dtype=torch.long)    \
                                                                                for c in input_columns[:-1]]
                self.tensors.append(torch.tensor(data[input_columns[-1]], dtype=torch.long))
                
            # unsupervised dataset
            elif d_type == 'unsup':
                input_columns = ['ori_input_ids', 'ori_input_type_ids', 'ori_input_mask',
                                 'aug_input_ids', 'aug_input_type_ids', 'aug_input_mask']
                self.tensors = [torch.tensor(data[c].apply(lambda x: ast.literal_eval(x)), dtype=torch.long)    \
                                                                                for c in input_columns]
                
            else:
                raise "d_type error. (d_type have to sup or unsup)"

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def get_sup(self, lines):
        raise NotImplementedError

    def get_unsup(self, lines):
        raise NotImplementedError

class pre_pipeline():
    def __init__(self,preprocessor,tokenize,indexer,labels,max_len=512):
        super().__init__()
        self.preprocessor = preprocessor
        self.tokenize = tokenize
        self.max_len = max_len


        self.indexer = indexer # function : tokens to indexes
        # map from a label name to a label index
        self.label_map = {name: i for i, name in enumerate(labels)}
        self.max_len = max_len

    def __call__(self, instance,d_type):

        ### Tokenizing ###
        label, text_a, text_b = instance
        
        label = self.preprocessor(label) if label else None
        tokens_a = self.tokenize(self.preprocessor(text_a))
        tokens_b = self.tokenize(self.preprocessor(text_b)) if text_b else []

        instance = (label, tokens_a, tokens_b)


        ### AddSpecialTokensWithTruncation ###
        label, tokens_a, tokens_b = instance

        # -3 special tokens for [CLS] text_a [SEP] text_b [SEP]
        # -2 special tokens for [CLS] text_a [SEP]
        _max_len = self.max_len - 3 if tokens_b else self.max_len - 2
        truncate_tokens_pair(tokens_a, tokens_b, _max_len)

        # Add Special Tokens
        tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        tokens_b = tokens_b + ['[SEP]'] if tokens_b else []

        ### TokenIndexing ###

        label, tokens_a, tokens_b = instance

        input_ids = self.indexer(tokens_a + tokens_b)
        segment_ids = [0]*len(tokens_a) + [1]*len(tokens_b) # type_ids
        input_mask = [1]*(len(tokens_a) + len(tokens_b))
        label_id = self.label_map[label] if label else None

        # zero padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)

        if label_id != None:
            return (input_ids, segment_ids, input_mask, label_id)
        else:
            return (input_ids, segment_ids, input_mask)

def dataset_class(args):
    table = {'imdb': IMDB,
             'dbp':DBP,
             'yelp_5':Yelp_5,
             'amz_2':Amazon_2,
             'yelp_2':Yelp_2,
             'amz_5':Amazon_5}

    return table[args.task]

### 5 different types of dataset ###
class IMDB(CsvDataset):
    labels = ('0', '1')
    def __init__(self, file, need_prepro, pipeline=[], max_len=128, mode='train', d_type='sup',cfg=None,args=None):
        super().__init__(file, need_prepro, pipeline, max_len, mode, d_type,cfg,args)

    def get_sup(self, lines):
        for line in itertools.islice(lines, 0, None):
            yield line[7], line[6], []    # label, text_a, None
            # yield None, line[6], []

    def get_unsup(self, lines):
        for line in itertools.islice(lines, 0, None):
            yield (None, line[1], []), (None, line[2], [])  # ko, en

class DBP(CsvDataset):
    labels = ('0', '1','2','3','4','5','6','7','8','9','10','11','12','13')
    def __init__(self, file, need_prepro, pipeline=[], max_len=128, mode='train', d_type='sup',cfg=None,args=None):
        super().__init__(file, need_prepro, pipeline, max_len, mode, d_type,cfg,args)

    def get_sup(self, lines):
        for line in itertools.islice(lines, 0, None):
            yield line[7], line[6], []    # label, text_a, None
            # yield None, line[6], []

    def get_unsup(self, lines):
        for line in itertools.islice(lines, 0, None):
            yield (None, line[1], []), (None, line[2], [])  # ko, en

class Yelp_5(CsvDataset):
    labels = ('0', '1','2','3','4')
    def __init__(self, file, need_prepro, pipeline=[], max_len=128, mode='train', d_type='sup',cfg=None,args=None):
        super().__init__(file, need_prepro, pipeline, max_len, mode, d_type,cfg,args)

    def get_sup(self, lines):
        for line in itertools.islice(lines, 0, None):
            yield line[7], line[6], []    # label, text_a, None
            # yield None, line[6], []

    def get_unsup(self, lines):
        for line in itertools.islice(lines, 0, None):
            yield (None, line[1], []), (None, line[2], [])  # ko, en

class Amazon_2(CsvDataset):
    labels = ('0', '1')
    def __init__(self, file, need_prepro, pipeline=[], max_len=128, mode='train', d_type='sup',cfg=None,args=None):
        super().__init__(file, need_prepro, pipeline, max_len, mode, d_type,cfg,args)

    def get_sup(self, lines):
        for line in itertools.islice(lines, 0, None):
            yield line[7], line[6], []    # label, text_a, None
            # yield None, line[6], []

    def get_unsup(self, lines):
        for line in itertools.islice(lines, 0, None):
            yield (None, line[1], []), (None, line[2], [])  # ko, en

class Yelp_2(CsvDataset):
    labels = ('0', '1')
    def __init__(self, file, need_prepro, pipeline=[], max_len=128, mode='train', d_type='sup',cfg=None,args=None):
        super().__init__(file, need_prepro, pipeline, max_len, mode, d_type,cfg,args)

    def get_sup(self, lines):
        for line in itertools.islice(lines, 0, None):
            yield line[7], line[6], []    # label, text_a, None
            # yield None, line[6], []

    def get_unsup(self, lines):
        for line in itertools.islice(lines, 0, None):
            yield (None, line[1], []), (None, line[2], [])  # ko, en

class Amazon_5(CsvDataset):
    labels = ('0', '1','2','3','4')
    def __init__(self, file, need_prepro, pipeline=[], max_len=128, mode='train', d_type='sup',cfg=None,args=None):
        super().__init__(file, need_prepro, pipeline, max_len, mode, d_type,cfg,args)

    def get_sup(self, lines):
        for line in itertools.islice(lines, 0, None):
            yield line[7], line[6], []    # label, text_a, None
            # yield None, line[6], []

    def get_unsup(self, lines):
        for line in itertools.islice(lines, 0, None):
            yield (None, line[1], []), (None, line[2], [])  # ko, en

class load_data:
    def __init__(self, cfg,args):
        self.cfg = cfg
        self.args = args

        self.TaskDataset = dataset_class(args)
        self.pipeline = None
        if cfg.need_prepro:
            tokenizer = tokenization.FullTokenizer(vocab_file=cfg.vocab, do_lower_case=cfg.do_lower_case)
            self.pipeline = [Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
                        AddSpecialTokensWithTruncation(cfg.max_seq_length),
                        TokenIndexing(tokenizer.convert_tokens_to_ids, self.TaskDataset.labels, cfg.max_seq_length)]
        
        if cfg.mode == 'train':
            self.sup_data_dir = cfg.sup_data_dir
            self.sup_batch_size = cfg.train_batch_size
            self.shuffle = True
        elif cfg.mode == 'train_eval':
            self.sup_data_dir = cfg.sup_data_dir
            self.eval_data_dir= cfg.eval_data_dir
            self.sup_batch_size = cfg.train_batch_size
            self.eval_batch_size = cfg.eval_batch_size
            self.shuffle = True
        elif cfg.mode == 'eval':
            self.sup_data_dir = cfg.eval_data_dir
            self.sup_batch_size = cfg.eval_batch_size
            self.shuffle = False                            # Not shuffel when eval mode
        
        if cfg.uda_mode:                                    # Only uda_mode
            self.unsup_data_dir = cfg.unsup_data_dir
            self.unsup_batch_size = cfg.train_batch_size * cfg.unsup_ratio

    def sup_data_iter(self):
        sup_dataset = self.TaskDataset(self.sup_data_dir, self.cfg.need_prepro, self.pipeline, self.cfg.max_seq_length, self.cfg.mode, 'sup', self.cfg,self.args)
        sup_data_iter = DataLoader(sup_dataset, batch_size=self.sup_batch_size, shuffle=self.shuffle)
        
        return sup_data_iter

    def unsup_data_iter(self):
        unsup_dataset = self.TaskDataset(self.unsup_data_dir, self.cfg.need_prepro, self.pipeline, self.cfg.max_seq_length, self.cfg.mode, 'unsup',self.cfg,self.args)
        unsup_data_iter = DataLoader(unsup_dataset, batch_size=self.unsup_batch_size, shuffle=self.shuffle)

        return unsup_data_iter

    def eval_data_iter(self):
        eval_dataset = self.TaskDataset(self.eval_data_dir, self.cfg.need_prepro, self.pipeline, self.cfg.max_seq_length, 'eval', 'sup',self.cfg,self.args)
        eval_data_iter = DataLoader(eval_dataset, batch_size=self.eval_batch_size, shuffle=False)

        return eval_data_iter
