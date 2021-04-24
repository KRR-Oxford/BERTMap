from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd

def load_onto_tsv_dataset(data_path, tokenizer: AutoTokenizer):
    na_vals = pd.io.parsers.STR_NA_VALUES.difference({'NULL', 'null', 'n/a'})
    df = pd.read_csv(data_path, sep='\t', na_values=na_vals, keep_default_na=False)
    dataset = Dataset.from_pandas(df)
    encoded_dataset = dataset.map(lambda examples: tokenizer(str(examples['Label1']), str(examples['Label2']),
                                                             truncation=True, padding=True), batched=True)
    encoded_dataset = encoded_dataset.map(lambda example: {'labels': example['Synonymous']}, 
                                          remove_columns=['Synonymous']) 
    encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])
    return encoded_dataset