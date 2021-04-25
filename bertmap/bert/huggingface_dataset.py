from transformers import AutoTokenizer
from datasets import Dataset
import torch
import pandas as pd

def load_onto_tsv_dataset(data_path, tokenizer: AutoTokenizer, batch_size=1024, max_length=512):
    na_vals = pd.io.parsers.STR_NA_VALUES.difference({'NULL', 'null', 'n/a'})
    df = pd.read_csv(data_path, sep='\t', na_values=na_vals, keep_default_na=False)
    dataset = Dataset.from_pandas(df)
    def encode(examples):
        item = tokenizer(examples['Label1'], examples['Label2'], return_tensors='pt', 
                         max_length=max_length, padding='longest', truncation=True)
        item['labels'] = torch.tensor(examples['Synonymous'])
        return item
    dataset.set_transform(encode)
    return dataset