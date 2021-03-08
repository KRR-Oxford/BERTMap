import sys
sys.path.append("/home/yuahe/projects/OntoAlign-py")
from ontoalign.experiments.direct_search import DirectBertExperiment
import pandas as pd

for name in ["mean", "cls"]:
    for src, tgt in [("fma", "nci"), ("fma", "snomed"), ("snomed", "nci")]:
        
        base = "/home/yuahe/projects/OntoAlign-py/largebio_data/onto_labels"
        src_onto_lexicon_tsv = f"{base}/{src}2{tgt}.small.labels.tsv"
        tgt_onto_lexicon_tsv = f"{base}/{tgt}2{src}.small.labels.tsv"
        src_embeds_pt = f"{base}/{src}2{tgt}.small.{name}.pt"
        tgt_embeds_pt = f"{base}/{tgt}2{src}.small.{name}.pt"
        
        exp = DirectBertExperiment(
                                src, tgt, 
                                src_onto_lexicon_tsv, tgt_onto_lexicon_tsv,
                                f"/home/yuahe/projects/OntoAlign-py/largebio_exp/small/{src}2{tgt}/",
                                src_embeds_pt, tgt_embeds_pt,
                                src_batch_size=10000, tgt_batch_size=10000,
                                task_suffix="small", name=f"bc-{name}", 
                                bert_path="emilyalsentzer/Bio_ClinicalBERT")
        exp.run()
        exp.save()
