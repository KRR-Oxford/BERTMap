
import os
main_dir = os.getcwd().split("BERTMap")[0] + "BERTMap"
import sys
sys.path.append(main_dir)
from bertmap.map import BERTEmbedsMapping
import pathlib


src = sys.argv[1]
tgt = sys.argv[2]
name = sys.argv[3]  # cls mean
label_dir = "/home/yuahe/projects/BERTMap/data/largebio/labels"
exp_base = f"{main_dir}/experiment/bert_baseline/{src}2{tgt}/{name}"
pathlib.Path(exp_base).mkdir(parents=True, exist_ok=True) 
        
src_onto_class2text_tsv = f"{label_dir}/{src}2{tgt}.small.labels.tsv"
tgt_onto_class2text_tsv = f"{label_dir}/{tgt}2{src}.small.labels.tsv"

bert_map = BERTEmbedsMapping(src, tgt, src_onto_class2text_tsv, tgt_onto_class2text_tsv, 
                                exp_base, batch_size=256, nbest=1, task_suffix="small", name=f"bc-{name}", 
                                bert_path="emilyalsentzer/Bio_ClinicalBERT", string_match=True)
bert_map.set_inverted_index("SRC")
bert_map.set_inverted_index("TGT")
bert_map.candidate_limit = int(sys.argv[4]) 
bert_map.run()

