import sys
sys.path.append("/home/yuahe/projects/BERTMap")
from bertmap.corpora import CrossOntoCorpus

base = "/home/yuahe/projects/BERTMap/data/largebio/"

for src, tgt in [("fma", "nci"), ("fma", "snomed"), ("snomed", "nci")]:
    src_onto = base + f"ontos/{src}2{tgt}.small.owl"
    src_labels = base + f"labels/{src}2{tgt}.small.labels.tsv"
    tgt_onto = base + f"ontos/{tgt}2{src}.small.owl"
    tgt_labels = base + f"labels/{tgt}2{src}.small.labels.tsv"
    ref = base + f"refs/{src}2{tgt}.legal.tsv"
    src2tgt = CrossOntoCorpus(f"{src}2{tgt}", src_onto, tgt_onto, ref, src_labels, tgt_labels, sample_rate=5)
    src2tgt.save_corpus(base + "/corpora")
    # src2tgt.load_corpus(base + "/corpora")