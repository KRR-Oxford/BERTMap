# It doesn't work with Huggingface Trainer for now ...

from transformers import AutoTokenizer
from datasets import Dataset
import torch
import pandas as pd

def load_onto_tsv_dataset(data_tsv, tokenizer: AutoTokenizer, max_length=512):
    na_vals = pd.io.parsers.STR_NA_VALUES.difference({'NULL', 'null', 'n/a'})
    df = pd.read_csv(data_tsv, sep='\t', na_values=na_vals, keep_default_na=False).\
        rename(columns={"Synonymous": "labels"})
    dataset = Dataset.from_pandas(df)
    def encode(examples):
        items = tokenizer(examples['Label1'], examples['Label2'], return_tensors='pt', 
                        max_length=max_length, padding='max_length', truncation=True)
        items['labels'] = torch.tensor(examples['labels'])
        return items
    dataset.set_transform(encode)
    return dataset