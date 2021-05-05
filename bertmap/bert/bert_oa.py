"""
Fine-tuning BERT with the class-text datasets from ontologies
Code inspired by: https://huggingface.co/transformers/training.html
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score
from typing import List
from datasets import Dataset
import json
import pandas as pd

class BERTOntoAlign:
    
    def __init__(self, 
                 bert_checkpoint: str, 
                 data_file: str, 
                 training_args: TrainingArguments, 
                 early_stop: bool=True):
        print(f"initialize BERT for Binary Classification from the Pretrained BERT model at: {bert_checkpoint} ...")
        
        # BERT
        self.model = AutoModelForSequenceClassification.from_pretrained(bert_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint)

        # data
        with open(data_file, "r") as f: oa_data = json.load(f)
        self.train = self.load_dataset(oa_data["train"])
        self.val = self.load_dataset(oa_data["val"])
        self.test = self.load_dataset(oa_data["test"])
        print(f"data files loaded with sizes:\n\t[# Train]: {len(self.train)}, [# Val]: {len(self.val)}, [# Test]: {len(self.test)}")
        
        # trainer
        self.training_args = training_args
        self.trainer = Trainer(model=self.model, args=self.training_args, 
                               train_dataset=self.train, eval_dataset=self.val, 
                               compute_metrics=self.compute_metrics, tokenizer=self.tokenizer)
        if early_stop: self.trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))
        
    @staticmethod
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc}
    
    def load_dataset(self, data: List[List], batch_size: int=1024, max_length: int=512):
        data_df = pd.DataFrame(data, columns=["sent1", "sent2", "labels"])
        dataset = Dataset.from_pandas(data_df)
        dataset = dataset.map(lambda examples: 
            self.tokenizer(examples['sent1'], examples['sent2'], max_length=max_length, truncation=True), 
            batched=True, batch_size=batch_size, num_proc=10)
        return dataset
