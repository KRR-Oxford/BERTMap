"""Direct Search Experiment on using the normalized edit distance score as the distance metric.
"""

from onto_align.onto import OntoMetric, Ontology
from onto_align.designed_exp import DirectSearchExperiment
import random

class NormEditSimExperiment(DirectSearchExperiment):
    
    def __init__(self, src_onto_iri_abbr, tgt_onto_iri_abbr, src_data_tsv, tgt_data_tsv, task_suffix="small"):
        super().__init__(src_onto_iri_abbr, tgt_onto_iri_abbr, src_data_tsv, tgt_data_tsv, task_suffix=task_suffix, exp_name="norm_edit_dist")
            
    def batch_mappings(self, batch_inds, queue=None, inverse=False):
        """Generate a batch of mappings for given source or target (inverse=True) entity batch indices"""
        batch_dict = dict()
        src_tsv = self.src_tsv if not inverse else self.tgt_tsv
        tgt_tsv = self.tgt_tsv if not inverse else self.src_tsv
        
        for i in batch_inds:
            src_row = src_tsv.iloc[i]
            src_entity_iri = src_row["entity-iri"]
            src_labels = src_row["entity-labels-list"].split(" <sep> ")
            min_dist = 1
            tgt_entity_iri = None
            
            for j in range(len(tgt_tsv)):
                tgt_row = tgt_tsv.iloc[j]
                tgt_labels = tgt_row["entity-labels-list"].split(" <sep> ")
                entity_dist = OntoMetric.min_norm_edit_dist(src_labels, tgt_labels)
                if (entity_dist < min_dist) or (entity_dist == min_dist and random.random() < 0.5):
                    min_dist = entity_dist
                    tgt_entity_iri = tgt_row["entity-iri"]   
            
            if not inverse:
                batch_dict[i] = [Ontology.reformat_entity_uri(src_entity_iri, self.src_iri), 
                                 Ontology.reformat_entity_uri(tgt_entity_iri, self.tgt_iri), 
                                 1 - min_dist]
            else:
                batch_dict[i] = [Ontology.reformat_entity_uri(tgt_entity_iri, self.src_iri), 
                                 Ontology.reformat_entity_uri(src_entity_iri, self.tgt_iri), 
                                 1 - min_dist]               
        
        if queue:
            queue.put(batch_dict)
            
        return batch_dict
