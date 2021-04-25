from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import torch

class OntoLabelDataset(Dataset):

    def __init__(self, data_tsv, tokenizer: AutoTokenizer, max_length=512):
        # Model parameter
        na_vals = pd.io.parsers.STR_NA_VALUES.difference({'NULL', 'null', 'n/a'})
        data = pd.read_csv(data_tsv, sep='\t', na_values=na_vals, keep_default_na=False)
        self.labels = data["Synonymous"]
        self.encodings = tokenizer(list(zip(data["Label1"], data["Label2"])), 
                                   padding=True, max_length=max_length, truncation=True)  
        # truncation is no need as there is no long sentence here

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
