from onto_align.designed_exp import DirectSearchExperiment
from onto_align.onto import OntoMetric, Ontology
from onto_align.word_embed import PretrainedBert
import random


class DirectBertExperiment(DirectSearchExperiment):
    
    def __init__(self, src_onto_iri_abbr, tgt_onto_iri_abbr, 
                 src_data_tsv, tgt_data_tsv, save_path, 
                 task_suffix="small", bert_name="bio_clinical_bert", 
                 bert_path="emilyalsentzer/Bio_ClinicalBERT", bert_pooling_strategy="last-2-mean"):
        super().__init__(src_onto_iri_abbr, tgt_onto_iri_abbr, src_data_tsv, tgt_data_tsv, save_path, 
                         task_suffix=task_suffix, exp_name=bert_name)
        self.bert = PretrainedBert(bert_path)
        self.pooling_strategy = bert_pooling_strategy
    
    def lexicon_process(self, entity_lexicon):
        # strategy "last-2-mean" or "last-1-cls"
        entity_lexicon = entity_lexicon.replace(" <property> ", " ").replace(" <sep> ", " ")
        lexicon_embed = self.bert.get_basic_sent_embeddings(entity_lexicon, strategy=self.pooling_strategy)
        return lexicon_embed
    
    def entity_dist_metric(self, src_lexicon, tgt_lexicon):
        return OntoMetric.cos_dist(src_lexicon, tgt_lexicon)
