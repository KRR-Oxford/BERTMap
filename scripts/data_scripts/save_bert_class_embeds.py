import sys
sys.path.append("/home/yuahe/projects/BERTMap")
from bertmap.bert import BERTEmbeddings
from bertmap.bert import BERTClassEmbedding
import torch

label_dir = "/home/yuahe/projects/BERTMap/data/largebio/labels"
save_dir = "/home/yuahe/projects/BERTMap/experiment/bert_baseline/class_embeds"

bert = BERTEmbeddings("emilyalsentzer/Bio_ClinicalBERT")
bert_cls_embed = BERTClassEmbedding(bert, neg_layer_num=-1)
choice = "small"

# small
if choice == "small":
    for suffix in ["mean", "cls"]:
        
        bert_method = "batch_sent_embeds_" + suffix
        name = suffix

        for src, tgt in [("fma", "nci"), ("fma", "snomed"), ("snomed", "nci")]:
            
            src_data = f"{label_dir}/{src}2{tgt}.small.labels.tsv"
            tgt_data = f"{label_dir}/{tgt}2{src}.small.labels.tsv"
            
            src_ent_embeds = bert_cls_embed.class_embeds_from_ontology(bert_method, src_data, batch_size=1000)
            torch.save(src_ent_embeds, f"{save_dir}/{src}2{tgt}.small.{name}.pt")
            
            tgt_ent_embeds = bert_cls_embed.class_embeds_from_ontology(bert_method, tgt_data, batch_size=1000)
            torch.save(tgt_ent_embeds, f"{save_dir}/{tgt}2{src}.small.{name}.pt")

# whole
if choice == "whole":
    for suffix in ["mean", "cls"]:
        
        bert_method = "batch_sent_embeds_" + suffix
        name = suffix.split("_")[2]

        for onto in ["fma", "nci", "snomed"]:
            
            onto_data = f"{label_dir}/{onto}.whole.labels.tsv"
            
            ent_embeds = bert_cls_embed.class_embeds_from_ontology(bert_method, onto_data, batch_size=2000)
            torch.save(ent_embeds, f"{save_dir}/{onto}.whole.{name}.pt")
