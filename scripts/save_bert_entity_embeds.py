import sys
sys.path.append("/home/yuahe/projects/OntoAlign-py")
from ontoalign.embeds import PretrainedBert
from ontoalign.embeds import BertEntityEmbedding
import torch
base = "/home/yuahe/projects/OntoAlign-py/largebio_data/onto_labels"

bert = PretrainedBert("emilyalsentzer/Bio_ClinicalBERT")
bert_ent = BertEntityEmbedding(bert)
bert_method_suffix = "last_2_mean"
bert_method = "batch_sent_embeds_" + bert_method_suffix

for src, tgt in [("fma", "nci"), ("fma", "snomed"), ("snomed", "nci")]:
    
    src_data = f"{base}/{src}2{tgt}.small.labels.tsv"
    tgt_data = f"{base}/{tgt}2{src}.small.labels.tsv"
    
    
    src_ent_embeds = bert_ent.entity_embeds_from_ontology(bert_method, src_data, batch_size=1000)
    tgt_ent_embeds = bert_ent.entity_embeds_from_ontology(bert_method, tgt_data, batch_size=1000)
    
    torch.save(src_ent_embeds, f"{base}/{src}2{tgt}.small.embeds.pt")
    torch.save(tgt_ent_embeds, f"{base}/{tgt}2{src}.small.embeds.pt")
