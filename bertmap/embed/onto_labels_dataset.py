from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import torch

class OntoLabelsDataset(Dataset):

    def __init__(self, data_tsv, tokenizer: AutoTokenizer):
        # Model parameter
        data = pd.read_csv(data_tsv, sep="\t")
        text_pairs = []
        self.labels = []
        for _, dp in data.iterrows():
            text_pairs.append([dp["Label1"], dp["Label2"]])
            self.labels.append(dp["Synonymous"])
        self.encodings = tokenizer(text_pairs, truncation=True, padding=True)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
