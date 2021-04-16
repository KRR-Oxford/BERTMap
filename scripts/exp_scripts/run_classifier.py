main_dir = "/home/lawhy0729/BERTMap"
import sys
sys.path.append(main_dir)
import pandas as pd
from bertmap.bert import PretrainedBERT
from bertmap.onto import Ontology, OntoEvaluator
from bertmap.map.direct_search import DirectBERTClassifierMapping
from bertmap.utils import get_device
from bertmap.corpora import CrossOntoCorpus
import torch
import time
 
torch.cuda.empty_cache()

task_dict = {"ss": "semi-supervised", "us": "unsupervised"}
# configurations
src = "fma"
tgt = "nci"
task_abbr = "us"
task = task_dict[task_abbr]
setting = "f"
best_ckp = 37000
ckp_base = f"{main_dir}/experiment/bert_fine_tune/check_points/{task}/{src}2{tgt}.{task_abbr}.{setting}/checkpoint-{best_ckp}"

src_label_path = main_dir + f"/data/largebio/labels/{src}2{tgt}.small.labels.tsv"
src_label_path = main_dir + f"/{src}_labels_from_maps.tsv"
tgt_label_path = main_dir + f"/data/largebio/labels/{tgt}2{src}.small.labels.tsv"

# pre_bert = PretrainedBERT(pretrained_path=ckp_base, tokenizer_path="emilyalsentzer/Bio_ClinicalBERT", fine_tuned=False)
bert_map = DirectBERTClassifierMapping(src, tgt, src_label_path, tgt_label_path, 
                                       save_path=ckp_base + "/../trial", batch_size=3, 
                                       nbest=2, task_suffix="small", name="bc-tuned-mean", 
                                       bert_path=ckp_base, tokenizer_path="emilyalsentzer/Bio_ClinicalBERT")

bert_map.batch_size = 32
bert_map.strategy = "mean"
bert_map.fixed_one_side_alignment(flag="SRC")
bert_map.src2tgt_mappings.to_csv(main_dir + f"/{src}2{tgt}_mappings.tsv")