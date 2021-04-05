"""
Pretrained BERT and its variants from Pytorch-based Huggingface Library.
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd


class BinaryBERT:
    
    def __init__(self, pretrained_bert_path, train_path, val_path, test_path):
        print("Initialize BERT for Binary Classification from the Pretrained BERT model...")
        
        # BERT
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_path)
        
        # data
        self.train = pd.read_csv(train_path, sep='\t')
        self.val = pd.read_csv(val_path, sep='\t')
        self.test = pd.read_csv(test_path, sep='\t')
        
    
    def data_iterator(self):
        pass
        # # Model parameter
        # MAX_SEQ_LEN = 128
        # PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        # UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

        # # Fields

        # label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
        # text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
        #                 fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
        # fields = [('label', label_field), ('title', text_field), ('text', text_field), ('titletext', text_field)]

        # # TabularDataset

        # train, valid, test = TabularDataset.splits(path=source_folder, train='train.csv', validation='valid.csv',
        #                                         test='test.csv', format='CSV', fields=fields, skip_header=True)

        # # Iterators

        # train_iter = BucketIterator(train, batch_size=16, sort_key=lambda x: len(x.text),
        #                             device=device, train=True, sort=True, sort_within_batch=True)
        # valid_iter = BucketIterator(valid, batch_size=16, sort_key=lambda x: len(x.text),
        #                             device=device, train=True, sort=True, sort_within_batch=True)
        # test_iter = Iterator(test, batch_size=16, device=device, train=False, shuffle=False, sort=False)