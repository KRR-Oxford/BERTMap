from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd

def load_onto_tsv_dataset(data_path, tokenizer: AutoTokenizer):
    df = pd.read_csv(data_path, sep='\t')
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda example: {'labels': [example['Synonymous']]}, remove_columns=['Synonymous'])  # example becomes "inputs": [sent 1, sent 2], "labels": 0 or 1
    encoded_dataset = dataset.map(lambda examples: tokenizer([str(example['Label1']), str(example['Label2'])]
                                                             , padding=True, max_length=512, truncation=True), batched=True)
    return encoded_dataset