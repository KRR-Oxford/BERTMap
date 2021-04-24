from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd

def load_onto_tsv_dataset(data_path, tokenizer: AutoTokenizer):
    df = pd.read_csv(data_path, sep='\t')
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda example: {'label': example['Synonymous']}, remove_columns=['Synonymous']) 
    encoded_dataset = dataset.map(lambda examples: tokenizer(str(examples['Label1']), str(examples['Label2']), 
                                                             padding=True, max_length=512, truncation=True), batched=True)
    return encoded_dataset