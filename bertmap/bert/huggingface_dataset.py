# It doesn't work with Huggingface Trainer for now ...

from transformers import AutoTokenizer
from datasets import Dataset
import torch
import pandas as pd

def load_onto_tsv_dataset(data_path, tokenizer: AutoTokenizer, max_length=512):
    na_vals = pd.io.parsers.STR_NA_VALUES.difference({'NULL', 'null', 'n/a'})
    df = pd.read_csv(data_path, sep='\t', na_values=na_vals, keep_default_na=False).\
        rename(columns={"Label1": "sentence1", "Label2": "sentence2", "Synonymous": "labels"})
    dataset = Dataset.from_pandas(df)
    def encode(examples):
        items = tokenizer(examples['sentence1'], examples['sentence1'], return_tensors='pt', 
                        max_length=max_length, padding='max_length', truncation=True)
        items['labels'] = torch.tensor(examples['labels'])
        return items
    dataset.set_transform(encode)
    return dataset