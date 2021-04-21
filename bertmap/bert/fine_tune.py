"""
Fine-tuning BERT with ontology labels datasets
Code inspired by: https://huggingface.co/transformers/training.html
"""
from os import stat
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from bertmap.bert import OntoLabelDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Optional


class OntoLabelBERT:
    
    def __init__(self, pretrained_bert_path, train_path, val_path, test_path, training_args: TrainingArguments, early_stop=True):
        print("Initialize BERT for Binary Classification from the Pretrained BERT model...")
        
        # BERT
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_path)

        # data
        self.train = OntoLabelDataset(train_path, self.tokenizer)
        self.val = OntoLabelDataset(val_path, self.tokenizer)
        self.test = OntoLabelDataset(test_path, self.tokenizer)
        print(f"[# Train]: {len(self.train)}, [# Val]: {len(self.val)}, [# Test]: {len(self.test)}")
        
        # trainer
        self.training_args = training_args
        self.trainer = Trainer(model=self.model, args=self.training_args, 
                               train_dataset=self.train, eval_dataset=self.val, 
                               compute_metrics=self.compute_metrics)
        if early_stop:
            self.trainer.add_callback(MyEarlyStoppingCallback(early_stopping_patience=5))
        
    
    @staticmethod
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        # precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            # 'f1': f1,
            # 'precision': precision,
            # 'recall': recall
        }
        
        
# Inherit the callback class to print the early stopping states
class MyEarlyStoppingCallback(EarlyStoppingCallback):
    
    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: Optional[float] = 0.0):
        super().__init__(early_stopping_patience=early_stopping_patience, early_stopping_threshold=early_stopping_threshold)
                
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        super().on_evaluate(args, state, control, metrics, **kwargs)
        print(f"\n[Early stopping status]: {self.early_stopping_patience_counter}/{self.early_stopping_patience}")