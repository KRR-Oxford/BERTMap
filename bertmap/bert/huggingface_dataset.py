from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd

def load_onto_tsv_dataset(data_path, tokenizer: AutoTokenizer, batch_size=1024, max_length=512):
    na_vals = pd.io.parsers.STR_NA_VALUES.difference({'NULL', 'null', 'n/a'})
    df = pd.read_csv(data_path, sep='\t', na_values=na_vals, keep_default_na=False)
    df = df.rename(columns={'Synonymous': 'labels'})
    dataset = Dataset.from_pandas(df)
    dataset.set_transform(lambda examples: 
        tokenizer(examples['Label1'], examples['Label2'], max_length=max_length, padding='longest', truncation=True))
    return dataset