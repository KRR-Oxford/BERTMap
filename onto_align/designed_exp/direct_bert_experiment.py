from onto_align.designed_exp import DirectSearchExperiment
from onto_align.onto import OntoMetric, Ontology
from onto_align.word_embed import PretrainedBert
import random
import torch


class DirectBertExperiment(DirectSearchExperiment):
    
    def __init__(self, src_onto_iri_abbr, tgt_onto_iri_abbr, 
                 src_data_tsv, tgt_data_tsv, save_path, 
                 src_embeds_pt, tgt_embeds_pt,
                 task_suffix="small", bert_name="bio_clinical_bert", 
                 bert_path="emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__(src_onto_iri_abbr, tgt_onto_iri_abbr, src_data_tsv, tgt_data_tsv, save_path, 
                         task_suffix=task_suffix, exp_name=bert_name)
        self.bert = PretrainedBert(bert_path)
        self.src_embeds = torch.load(src_embeds_pt)
        self.tgt_embeds = torch.load(tgt_embeds_pt)
    
    def lexicon_process(self, entity_id, flag):
        # strategy "last-2-mean" or "last-1-cls"
        embeds = self.src_embeds if flag == "SRC" else None
        embeds = self.tgt_embeds if flag == "TGT" else None
        return embeds[entity_id]

    def entity_dist_metric(self, src_lexicon, tgt_lexicon):
        return OntoMetric.cos_dist(src_lexicon, tgt_lexicon)
