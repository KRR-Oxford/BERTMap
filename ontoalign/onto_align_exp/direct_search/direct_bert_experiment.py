from onto_align.designed_exp import DirectSearchExperiment
from onto_align.onto import OntoMetric
from onto_align.word_embed import PretrainedBert
import torch
import os


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
        
    def batch_mappings(self, batch_inds, inverse=False):
        
        flag = "Fixed-SRC" if not inverse else "Fixed-TGT"
        pid = os.getpid()
        size = len(batch_inds)
        self.log_print(f"[Process {pid}][{flag}] Starting a batch with size {size}")
        
        batch_dict = dict()
        src_tsv = self.src_tsv if not inverse else self.tgt_tsv
        tgt_tsv = self.tgt_tsv if not inverse else self.src_tsv
        src_embeds = self.src_embeds if not inverse else self.tgt_embeds
        tgt_embeds = self.tgt_embeds if not inverse else self.src_embeds
        
        cos_matrix = OntoMetric.pairwise_cos_sim(src_embeds[batch_inds, :], tgt_embeds)  # (batch_size, target_full_size)
        assert cos_matrix.shape == (len(batch_inds), len(tgt_tsv))
        
        argmax_sim = list(cos_matrix.argmax(axis=1)) # (batch_size, )
        max_sim = list(cos_matrix.max(axis=1))  # (batch_size, )
        assert len(max_sim) == len(batch_inds) and len(argmax_sim) == len(batch_inds)
        
        for i in range(len(batch_inds)):
            src_row = src_tsv.iloc[batch_inds[i]]
            tgt_row = tgt_tsv.iloc[argmax_sim[i]]
            src_entity_iri = src_row["entity-iri"]
            tgt_entity_iri = tgt_row["entity-iri"]

            if not inverse:
                batch_dict[i] = [Ontology.reformat_entity_uri(src_entity_iri, self.src_iri), 
                                 Ontology.reformat_entity_uri(tgt_entity_iri, self.tgt_iri), 
                                 max_sim[i]]
            else:
                batch_dict[i] = [Ontology.reformat_entity_uri(tgt_entity_iri, self.src_iri), 
                                 Ontology.reformat_entity_uri(src_entity_iri, self.tgt_iri), 
                                 max_sim[i]]     
            
            self.log_print(f"[Process {pid}][{flag}][Map {i}] {batch_dict[i][0]} | {batch_dict[i][1]}") 
            
        self.log_print(f"[Process {pid}][{flag}] Finishing the batch ...")
        
        return batch_dict
    
    def lexicon_process(self, _, entity_id, entity_flag):
        # strategy "last-2-mean" or "last-1-cls"
        embeds = self.src_embeds if entity_flag == "SRC" else None
        embeds = self.tgt_embeds if entity_flag == "TGT" else None
        return embeds[entity_id]

    def entity_dist_metric(self, src_lexicon, tgt_lexicon):
        return OntoMetric.cos_dist(src_lexicon, tgt_lexicon)
