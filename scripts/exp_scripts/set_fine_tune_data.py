import os
main_dir = os.getcwd().split("BERTMap")[0] + "BERTMap"
import sys
sys.path.append(main_dir)
from bertmap.corpora import CrossOntoCorpus, MergedOntoCorpus
from random import shuffle

corpora_dir = f"{main_dir}/data/largebio/corpora"
exp_dir = f"{main_dir}/experiment/bert_fine_tune/data"
src, tgt = "snomed", "nci"  # "fma", "nci"; "fma", "snomed"; "snomed", "nci"

cross_onto = CrossOntoCorpus(f"{src}2{tgt}", corpus_file=corpora_dir)
merged_intra_onto = MergedOntoCorpus(f"{src}2{tgt}.small", corpus_file=corpora_dir)

# Cross ontology level
test_r = cross_onto.train_val_test_split(only_test=True)
train_r, val_r, test_r = cross_onto.train_val_test_split(only_test=False)

cross_onto.save_semantic_pairs(test_r, exp_dir + f"/{src}2{tgt}.us.test.r.tsv")
cross_onto.save_semantic_pairs(train_r, exp_dir + f"/{src}2{tgt}.ss.train.r.tsv")
cross_onto.save_semantic_pairs(val_r, exp_dir + f"/{src}2{tgt}.ss.val.r.tsv")
cross_onto.save_semantic_pairs(test_r, exp_dir + f"/{src}2{tgt}.ss.test.r.tsv")

# Semi-supervised training data
train_f_r = merged_intra_onto.train_val_split(only_train=True) + train_r
shuffle(train_f_r)
train_f_b_r = merged_intra_onto.train_val_split(backward=True, only_train=True) + train_r
shuffle(train_f_b_r)
train_f_b_i_r = merged_intra_onto.train_val_split(include_ids=True, backward=True, only_train=True) + train_r
shuffle(train_f_b_i_r)

merged_intra_onto.save_semantic_pairs(train_f_r, exp_dir + f"/{src}2{tgt}.ss.train.f+r.tsv")
merged_intra_onto.save_semantic_pairs(train_f_b_r, exp_dir + f"/{src}2{tgt}.ss.train.f+b+r.tsv")
merged_intra_onto.save_semantic_pairs(train_f_b_i_r, exp_dir + f"/{src}2{tgt}.ss.train.f+b+i+r.tsv")

# Unsupervised training; validation data
train_f, val_f = merged_intra_onto.train_val_split(only_train=False)
train_f_b, val_f_b = merged_intra_onto.train_val_split(backward=True, only_train=False)
train_f_b_i, val_f_b_i = merged_intra_onto.train_val_split(include_ids=True, backward=True, only_train=False)

merged_intra_onto.save_semantic_pairs(train_f, exp_dir + f"/{src}2{tgt}.us.train.f.tsv")
merged_intra_onto.save_semantic_pairs(train_f_b, exp_dir + f"/{src}2{tgt}.us.train.f+b.tsv")
merged_intra_onto.save_semantic_pairs(train_f_b, exp_dir + f"/{src}2{tgt}.us.train.f+b+i.tsv")

merged_intra_onto.save_semantic_pairs(val_f, exp_dir + f"/{src}2{tgt}.us.val.f.tsv")
merged_intra_onto.save_semantic_pairs(val_f_b, exp_dir + f"/{src}2{tgt}.us.val.f+b.tsv")
merged_intra_onto.save_semantic_pairs(val_f_b, exp_dir + f"/{src}2{tgt}.us.val.f+b+i.tsv")