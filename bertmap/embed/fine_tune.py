"""
Fine-tuning BERT with ontology labels datasets
Code inspired by: https://huggingface.co/transformers/training.html
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from bertmap.embed import OntoLabelsDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class OntoLabelsBERT:
    
    def __init__(self, pretrained_bert_path, train_path, val_path, forward_test_path, backward_test_path, training_args: TrainingArguments):
        print("Initialize BERT for Binary Classification from the Pretrained BERT model...")
        
        # BERT
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_path)

        # data
        self.train = OntoLabelsDataset(train_path, self.tokenizer)
        self.val = OntoLabelsDataset(val_path, self.tokenizer)
        self.forward_test = OntoLabelsDataset(forward_test_path, self.tokenizer)
        self.backward_test = OntoLabelsDataset(backward_test_path, self.tokenizer)
        
        # trainer
        self.training_args = training_args
        self.trainer = Trainer(model=self.model, args=self.training_args, 
                               train_dataset=self.train, eval_dataset=self.val, 
                               compute_metrics=self.compute_metrics)
        
    
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