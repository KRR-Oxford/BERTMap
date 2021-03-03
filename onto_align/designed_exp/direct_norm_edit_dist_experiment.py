"""Direct Search Experiment on using the normalized edit distance score as the distance metric.
"""

from onto_align.onto import OntoMetric, Ontology
from onto_align.designed_exp import DirectSearchExperiment
import random

class DirectNormEditSimExperiment(DirectSearchExperiment):
    
    def __init__(self, src_onto_iri_abbr, tgt_onto_iri_abbr, src_data_tsv, tgt_data_tsv, task_suffix="small"):
        super().__init__(src_onto_iri_abbr, tgt_onto_iri_abbr, src_data_tsv, tgt_data_tsv, task_suffix=task_suffix, exp_name="norm_edit_dist")
        
    def lexicon_process(self, entity_lexicon):
        return entity_lexicon.split(" <sep> ")
        
    def entity_dist_metric(self, src_lexicon, tgt_lexicon):
        return OntoMetric.min_norm_edit_dist(src_lexicon, tgt_lexicon)
