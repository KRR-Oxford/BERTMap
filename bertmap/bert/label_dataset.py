from transformers import AutoTokenizer
import pandas as pd
import torch


class OntoLabelDataset(torch.utils.data.Dataset):

    def __init__(self, data_tsv, tokenizer: AutoTokenizer, max_length=512):
        # Model parameter
        self.data = pd.read_csv(data_tsv, sep="\t")
        self.labels = list(self.data["Synonymous"])
        self.encode = lambda examples: tokenizer(examples["Label1"], examples["Label2"], return_tensors='pt', 
                                                 max_length=max_length, padding="longest", truncation=True)
        # self.encodings = tokenizer(text_pairs, truncation=True, padding=True)  # truncation is no need as there is no long sentence here

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = self.encode(self.data.iloc[idx])
        # item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
