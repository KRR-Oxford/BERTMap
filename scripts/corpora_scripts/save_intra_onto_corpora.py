import sys
sys.path.append("/home/yuahe/projects/BERTMap")
from bertmap.corpora import IntraOntoCorpus, MergedOntoCorpus

base = "/home/yuahe/projects/BERTMap/data/largebio/"

for src, tgt in [("fma", "nci"), ("fma", "snomed"), ("snomed", "nci")]:
   src_onto = base + f"ontos/{src}2{tgt}.small.owl"
   src_labels = base + f"labels/{src}2{tgt}.small.labels.tsv"
   tgt_onto = base + f"ontos/{tgt}2{src}.small.owl"
   tgt_labels = base + f"labels/{tgt}2{src}.small.labels.tsv"

   oc_src = IntraOntoCorpus(src, src_onto, src_labels, sample_rate=6)
   oc_tgt = IntraOntoCorpus(tgt, tgt_onto, tgt_labels, sample_rate=6)
   oc_src2tgt = MergedOntoCorpus(f"{src}2{tgt}.small", oc_src, oc_tgt)
   oc_src2tgt.save_corpus("/home/yuahe/projects/BERTMap/data/largebio/corpora")
   # oc_src2tgt.load_corpus("/home/yuahe/projects/BERTMap/data/largebio/corpora")