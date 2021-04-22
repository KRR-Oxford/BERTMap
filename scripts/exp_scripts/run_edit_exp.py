import os
main_dir = os.getcwd().split("BERTMap")[0] + "BERTMap"
import sys
sys.path.append(main_dir)
from bertmap.map import NormEditSimMapping

label_dir = f"{main_dir}data/largebio/labels"

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


