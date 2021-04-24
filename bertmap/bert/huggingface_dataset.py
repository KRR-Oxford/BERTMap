from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd

def load_onto_tsv_dataset(data_path, tokenizer: AutoTokenizer):
    df = pd.read_csv(data_path, sep='\t')
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda example: {'labels': example['Synonymous']}, remove_columns=['Synonymous']) 
    encoded_dataset = dataset.map(lambda examples: tokenizer(str(examples['Label1']), str(examples['Label2']), 
                                                             truncation=True, padding=True), batched=True)
    encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])
    return encoded_dataset