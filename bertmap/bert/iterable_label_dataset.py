from transformers import AutoTokenizer
import pandas as pd
import torch


class OntoLabelDataset(torch.utils.data.IterableDataset):

    def __init__(self, data_tsv, tokenizer: AutoTokenizer):
        super().__init__()
        na_vals = pd.io.parsers.STR_NA_VALUES.difference({'NULL', 'null', 'n/a'})
        df = pd.read_csv(data_tsv, sep='\t', na_values=na_vals, keep_default_na=False)
        text_pairs = []
        # self.labels = []
        # for _, dp in data.iterrows():
        #     text_pairs.append([str(dp["Label1"]), str(dp["Label2"])])
        #     self.labels.append(dp["Synonymous"])
        # self.encodings = tokenizer(text_pairs, truncation=True, padding=True)  # truncation is no need as there is no long sentence here

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item