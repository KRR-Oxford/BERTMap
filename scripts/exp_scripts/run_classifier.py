main_dir = "/home/yuahe/projects/BERTMap"
import sys
sys.path.append(main_dir)
import pandas as pd
from bertmap.onto import Ontology, OntoEvaluator
from bertmap.map import BERTClassifierMapping
import torch
 
torch.cuda.empty_cache()

task_dict = {"ss": "semi-supervised", "us": "unsupervised"}
# configurations
src = sys.argv[1]
tgt = sys.argv[2]
task_abbr = sys.argv[3]  # us or ss
task = task_dict[task_abbr]
setting = sys.argv[4]  # f, f+b ...
best_ckp = 37000
ckp_base = f"{main_dir}/experiment/bert_fine_tune/{src}2{tgt}.{task_abbr}.{setting}/checkpoint-{best_ckp}"

src_label_path = main_dir + f"/data/largebio/labels/{src}2{tgt}.small.labels.tsv"
# src_label_path = main_dir + f"/{src}_labels_from_maps.tsv"
tgt_label_path = main_dir + f"/data/largebio/labels/{tgt}2{src}.small.labels.tsv"

# pre_bert = PretrainedBERT(pretrained_path=ckp_base, tokenizer_path="emilyalsentzer/Bio_ClinicalBERT", fine_tuned=False)
bert_map = BERTClassifierMapping(src, tgt, src_label_path, tgt_label_path, 
                                 save_path=ckp_base + "/..", batch_size=-1, 
                                 nbest=1, task_suffix="small", name="bc-tuned-mean", 
                                 bert_path=ckp_base, tokenizer_path="emilyalsentzer/Bio_ClinicalBERT", string_match=True)
bert_map.set_inverted_index("SRC")
bert_map.set_inverted_index("TGT")
bert_map.candidate_limit = 50
bert_map.batch_size = 100
bert_map.strategy = "mean"
bert_map.run()
