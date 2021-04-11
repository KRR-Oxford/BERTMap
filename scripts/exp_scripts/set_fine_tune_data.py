import sys
sys.path.append("/home/yuahe/projects/BERTMap")
from bertmap.corpora import CrossOntoCorpus, MergedOntoCorpus
from random import shuffle

corpora_dir = "/home/yuahe/projects/BERTMap/data/largebio/corpora"
exp_dir = "/home/yuahe/projects/BERTMap/experiment/bert_fine_tune/data"
src, tgt = "snomed", "nci"  # "fma", "nci"; "fma", "snomed"; "snomed", "nci"

cross_onto = CrossOntoCorpus(f"{src}2{tgt}", corpus_path=corpora_dir)
merged_intra_onto = MergedOntoCorpus(f"{src}2{tgt}.small", corpus_path=corpora_dir)

# Cross ontology level
test_r = cross_onto.train_val_test_split(only_test=True)
train_r, val_r, test_r = cross_onto.train_val_test_split(only_test=False)

cross_onto.save_labels(test_r, exp_dir + f"/unsupervised/{src}2{tgt}.us.test.r.tsv")
cross_onto.save_labels(train_r, exp_dir + f"/semi-supervised/{src}2{tgt}.ss.train.r.tsv")
cross_onto.save_labels(val_r, exp_dir + f"/semi-supervised/{src}2{tgt}.ss.val.r.tsv")
cross_onto.save_labels(test_r, exp_dir + f"/semi-supervised/{src}2{tgt}.ss.test.r.tsv")

# Semi-supervised training data
train_f_r = merged_intra_onto.train_val_split(only_train=True) + train_r
shuffle(train_f_r)
train_f_b_r = merged_intra_onto.train_val_split(backward=True, only_train=True) + train_r
shuffle(train_f_b_r)
train_f_b_i_r = merged_intra_onto.train_val_split(identity=True, backward=True, only_train=True) + train_r
shuffle(train_f_b_i_r)

merged_intra_onto.save_labels(train_f_r, exp_dir + f"/semi-supervised/{src}2{tgt}.ss.train.f+r.tsv")
merged_intra_onto.save_labels(train_f_b_r, exp_dir + f"/semi-supervised/{src}2{tgt}.ss.train.f+b+r.tsv")
merged_intra_onto.save_labels(train_f_b_i_r, exp_dir + f"/semi-supervised/{src}2{tgt}.ss.train.f+b+i+r.tsv")

# Unsupervised training; validation data
train_f, val_f = merged_intra_onto.train_val_split(only_train=False)
train_f_b, val_f_b = merged_intra_onto.train_val_split(backward=True, only_train=False)
train_f_b_i, val_f_b_i = merged_intra_onto.train_val_split(identity=True, backward=True, only_train=False)

merged_intra_onto.save_labels(train_f, exp_dir + f"/unsupervised/{src}2{tgt}.us.train.f.tsv")
merged_intra_onto.save_labels(train_f_b, exp_dir + f"/unsupervised/{src}2{tgt}.us.train.f+b.tsv")
merged_intra_onto.save_labels(train_f_b, exp_dir + f"/unsupervised/{src}2{tgt}.us.train.f+b+i.tsv")

merged_intra_onto.save_labels(val_f, exp_dir + f"/unsupervised/{src}2{tgt}.us.val.f.tsv")
merged_intra_onto.save_labels(val_f_b, exp_dir + f"/unsupervised/{src}2{tgt}.us.val.f+b.tsv")
merged_intra_onto.save_labels(val_f_b, exp_dir + f"/unsupervised/{src}2{tgt}.us.val.f+b+i.tsv")