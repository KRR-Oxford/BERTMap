from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import torch

class OntoLabelDataset(Dataset):

    def __init__(self, data_tsv, tokenizer: AutoTokenizer):
        # Model parameter
        data = pd.read_csv(data_tsv, sep="\t")
        text_pairs = []
        self.labels = []
        for _, dp in data.iterrows():
            text_pairs.append([str(dp["Label1"]), str(dp["Label2"])])
            self.labels.append(dp["Synonymous"])
        self.encode = lambda x: tokenizer(x, padding=True, max_length=512, truncation=True)
        self.set_transform(self.encode)

    def __len__(self):
        return len(self.labels)

    
