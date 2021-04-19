import sys
sys.path.append("/home/yuahe/projects/BERTMap")
from bertmap.map import NormEditSimMapping

label_dir = "/home/yuahe/projects/BERTMap/data/largebio/labels"
embed_dir = "/home/yuahe/projects/BERTMap/experiment/bert_baseline/class_embeds"

# ("fma", "nci"), ("fma", "snomed"), ("snomed", "nci")
for src, tgt in [("fma", "nci"), ("fma", "snomed"), ("snomed", "nci")]:
    
    src_onto_lexicon_tsv = f"{label_dir}/{src}2{tgt}.small.labels.tsv"
    tgt_onto_lexicon_tsv = f"{label_dir}/{tgt}2{src}.small.labels.tsv"
    
    exp = NormEditSimMapping(src, tgt, 
                           src_onto_lexicon_tsv, tgt_onto_lexicon_tsv,
                           f"/home/yuahe/projects/BERTMap/largebio_exp/small/{src}2{tgt}/",
                           task_suffix="small", num_pools=26)
    exp.run()
    exp.save()


