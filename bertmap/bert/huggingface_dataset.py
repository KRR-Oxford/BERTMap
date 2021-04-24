from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd

def load_onto_tsv_dataset(data_path, tokenizer: AutoTokenizer):
    df = pd.read_csv(data_path, sep='\t')
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda example: {'inputs': [example['Label1'], example['Label2']], 'labels': [example['Synonymous']]}, \
        remove_columns=['Label1', 'Label2', 'Synonymous'])  # example becomes "inputs": [sent 1, sent 2], "labels": 0 or 1
    encoded_dataset = dataset.map(lambda examples: tokenizer(examples['inputs']), batched=True)
    return encoded_dataset