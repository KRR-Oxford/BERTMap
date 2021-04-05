import sys
sys.path.append("/home/yuahe/projects/BERTMap")
from bertmap.corpora import IntraOntoCorpus
import pandas as pd

src = "fma"
tgt = "nci"
# corpus_names = ["forward_synonyms", "forward_soft_nonsynonyms", "forward_hard_nonsynonyms"]
corpus_names = ["forward_synonyms", "backward_synonyms", "forward_soft_nonsynonyms", "forward_hard_nonsynonyms"]
save_name = "f"

base = "/home/yuahe/projects/BERTMap/largebio_data/"
src_onto = base + f"ontologies/{src}2{tgt}.small.owl"
src_labels = base + f"onto_labels/{src}2{tgt}.small.labels.tsv"
tgt_onto = base + f"ontologies/{tgt}2{src}.small.owl"
tgt_labels = base + f"onto_labels/{tgt}2{src}.small.labels.tsv"
oc_src = IntraOntoCorpus(src_onto, src_labels, corpus_path=base + f"onto_corpora/{src}2{tgt}.small")
oc_tgt = IntraOntoCorpus(tgt_onto, tgt_labels, corpus_path=base + f"onto_corpora/{src}2{tgt}.small")

src_train, src_val = oc_src.train_val_split(corpus_names )
tgt_train, tgt_val = oc_tgt.train_val_split(corpus_names )

train = pd.concat([src_train, tgt_train], ignore_index=True).sample(frac=1).reset_index(drop=True)
val = pd.concat([src_val, tgt_val], ignore_index=True).sample(frac=1).reset_index(drop=True)

exp_base = f"/home/yuahe/projects/BERTMap/exp_fine_tune/small/{src}2{tgt}/data/"
train.to_csv(exp_base + f"train-{save_name}.tsv", index=False, sep="\t")
val.to_csv(exp_base + f"val-{save_name}.tsv", index=False, sep='\t')
