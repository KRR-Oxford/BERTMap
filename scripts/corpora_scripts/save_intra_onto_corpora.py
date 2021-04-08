import sys
sys.path.append("/home/yuahe/projects/BERTMap")
from bertmap.corpora import IntraOntoCorpus, IntraOntoCorpusPair

base = "/home/yuahe/projects/BERTMap/data/largebio/"

for src, tgt in [("fma", "nci"), ("fma", "snomed"), ("snomed", "nci")]:
   src_onto = base + f"ontos/{src}2{tgt}.small.owl"
   src_labels = base + f"labels/{src}2{tgt}.small.labels.tsv"
   tgt_onto = base + f"ontos/{tgt}2{src}.small.owl"
   tgt_labels = base + f"labels/{tgt}2{src}.small.labels.tsv"

   oc_src = IntraOntoCorpus(src_onto, src_labels, sample_rate=5)
   oc_tgt = IntraOntoCorpus(tgt_onto, tgt_labels, sample_rate=5)
   oc_src2tgt = IntraOntoCorpusPair(oc_src, oc_tgt, f"{src}2{tgt}.small")
   oc_src2tgt.save_corpus("/home/yuahe/projects/BERTMap/data/largebio/corpora")
   # oc_src2tgt.load_corpus("/home/yuahe/projects/BERTMap/data/largebio/corpora")