from transformers import AutoTokenizer
import pandas as pd
import torch


class OntoLabelDataset(torch.utils.data.Dataset):

    def __init__(self, data_tsv, tokenizer: AutoTokenizer, max_length=512):
        # Model parameter
        data = pd.read_csv(data_tsv, sep="\t")
        text_pairs = []
        self.labels = []
        for _, dp in data.iterrows():
            text_pairs.append([str(dp["Label1"]), str(dp["Label2"])])
            self.labels.append(dp["Synonymous"])
        self.encode = lambda x: tokenizer(x, return_tensors='pt', max_length=max_length, padding="longest", truncation=True)
        # self.encodings = tokenizer(text_pairs, truncation=True, padding=True)  # truncation is no need as there is no long sentence here

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = self.encode(self.text_pairs[idx])
        # item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
