import sys
sys.path.append("/home/yuahe/projects/BERTMap")
from bertmap.embed import OntoLabelsBERT
import pandas as pd
import torch
from bertmap.utils import get_device, set_seed

src = "fma"
tgt = "nci"
exp_base = f"/home/yuahe/projects/BERTMap/exp_fine_tune/small/{src}2{tgt}"
ref_base = "/home/yuahe/projects/BERTMap/largebio_data/ref_synonyms"
train_path = exp_base + "/data/train-f.tsv"
# train_path = exp_base + "/data/train-sample.tsv"
# val_path = exp_base + "/data/val-f.tsv"
# val_path = exp_base + "/data/val-sample.tsv"
for_test_path = ref_base + f"/{src}2{tgt}.forward_synonyms.tsv"
back_test_path = ref_base + f"/{src}2{tgt}.backward_synonyms.tsv"

# get 10% label pairs derived from reference mappings as the train-val set
ref_sample = pd.read_csv(for_test_path, sep="\t").append(pd.read_csv(back_test_path, sep="\t"), ignore_index=True).sample(frac=0.1)
data_sample = pd.read_csv(train_path, sep="\t").sample(n=len(ref_sample))
train_sample = ref_sample.append(data_sample, ignore_index=True).sample(frac=1.0)
val_sample = train_sample.sample(frac=0.2)
train_sample = train_sample.drop(index=val_sample.index)
train_sample.to_csv(exp_base + "/data/train-sample.tsv", sep='\t', index=False)
val_sample.to_csv(exp_base + "/data/val-sample.tsv", sep='\t', index=False)