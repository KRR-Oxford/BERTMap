import sys
sys.path.append("/home/yuahe/projects/BERTMap")
from bertmap.map.direct_search import DirectNESMapping

# ("fma", "nci"), ("fma", "snomed"), ("snomed", "nci")
for src, tgt in [("snomed", "nci")]:
    
    base = "/home/yuahe/projects/BERTMap/largebio_data/onto_labels"
    src_onto_lexicon_tsv = f"{base}/{src}2{tgt}.small.labels.tsv"
    tgt_onto_lexicon_tsv = f"{base}/{tgt}2{src}.small.labels.tsv"
    
    exp = DirectNESMapping(src, tgt, 
                           src_onto_lexicon_tsv, tgt_onto_lexicon_tsv,
                           f"/home/yuahe/projects/BERTMap/largebio_exp/small/{src}2{tgt}/",
                           task_suffix="small", num_pools=26)
    exp.run()
    exp.save()


