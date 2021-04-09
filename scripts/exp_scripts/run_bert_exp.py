import sys
sys.path.append("/home/yuahe/projects/BERTMap")
from bertmap.map.direct_search import DirectBERTMapping

label_dir = "/home/yuahe/projects/BERTMap/data/largebio/labels"
embed_dir = "/home/yuahe/projects/BERTMap/experiment/bert_baseline/class_embeds"


for name in ["mean", "cls"]:
    for src, tgt in [("fma", "nci"), ("fma", "snomed"), ("snomed", "nci")]:
        
        src_onto_class2text_tsv = f"{label_dir}/{src}2{tgt}.small.labels.tsv"
        tgt_onto_class2text_tsv = f"{label_dir}/{tgt}2{src}.small.labels.tsv"
        src_embeds_pt = f"{embed_dir}/{src}2{tgt}.small.{name}.pt"
        tgt_embeds_pt = f"{embed_dir}/{tgt}2{src}.small.{name}.pt"
        
        exp = DirectBERTMapping(
                                src, tgt, 
                                src_onto_class2text_tsv, tgt_onto_class2text_tsv,
                                embed_dir + "/../maps",
                                src_embeds_pt, tgt_embeds_pt,
                                src_batch_size=10000, tgt_batch_size=10000,
                                task_suffix="small", name=f"bc-{name}", 
                                bert_path="emilyalsentzer/Bio_ClinicalBERT")
        exp.run()
        exp.save()
