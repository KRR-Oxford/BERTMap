"""
Fine-tuning BERT with the classtext pair datasets extracted from ontologies
Code inspired by: https://huggingface.co/transformers/training.html
"""
from typing import List

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, Trainer, TrainingArguments)


class BERTTrainer:
    
    def __init__(self, 
                 bert_checkpoint: str, 
                 train_data: List,
                 val_data: List,
                 test_data: List,
                 max_length: int=128, 
                 early_stop: bool=False,
                 early_stop_patience: int=5):
        print(f"initialize BERT for Binary Classification from the Pretrained BERT model at: {bert_checkpoint} ...")
        
        # BERT
        self.model = AutoModelForSequenceClassification.from_pretrained(bert_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint)
        self.trainer = None

        # data
        self.tra = self.load_dataset(train_data, max_length=max_length)
        self.val = self.load_dataset(val_data, max_length=max_length)
        self.tst = self.load_dataset(test_data, max_length=max_length)
        print(f"text max length: {max_length}")
        print(f"data files loaded with sizes:")
        print(f"\t[# Train]: {len(self.tra)}, [# Val]: {len(self.val)}, [# Test]: {len(self.tst)}")
        
        # early stopping
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience

    def train(self, train_args: TrainingArguments):
        self.trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=self.tra,
            eval_dataset=self.val,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer
        )
        if self.early_stop: self.trainer.add_callback(
            EarlyStoppingCallback(early_stopping_patience=self.early_stop_patience)
        )
        self.trainer.train()

    @staticmethod
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc}
    
    def load_dataset(self, data: List, batch_size: int=1024, max_length: int=512) -> Dataset:
        data_df = pd.DataFrame(data, columns=["sent1", "sent2", "labels"])
        dataset = Dataset.from_pandas(data_df)
        dataset = dataset.map(
            lambda examples:
            self.tokenizer(
                examples['sent1'],
                examples['sent2'],
                max_length=max_length,
                truncation=True
            ),
            batched=True,
            batch_size=batch_size,
            num_proc=10
        )
        return dataset
