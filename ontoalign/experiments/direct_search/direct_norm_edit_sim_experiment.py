"""Direct Search Experiment on using the normalized edit distance score as the distance metric. 
   Unlike the BERT experiment where the batched (vectorization) algorithm is used, here we apply the multiprocessing on each batch.
"""

from ontoalign.onto import Ontology
from ontoalign.experiments.direct_search import DirectSearchExperiment
from itertools import product
from textdistance import levenshtein
from multiprocessing_on_dill import Pool
import os
import time

class DirectNormEditSimExperiment(DirectSearchExperiment):
    
    def __init__(self, src_onto_iri_abbr, tgt_onto_iri_abbr, 
                 src_onto_lexicon_tsv, tgt_onto_lexicon_tsv, save_path, 
                 task_suffix="small", num_pools=18):
        super().__init__(src_onto_iri_abbr, tgt_onto_iri_abbr, src_onto_lexicon_tsv, tgt_onto_lexicon_tsv, save_path, 
                         task_suffix=task_suffix, name="nes")
        
        self.num_pools = num_pools
        self.src_procs = []
        self.tgt_procs= []

    def alignment(self):
        pool = Pool(self.num_pools)
        self.src_procs = []
        self.tgt_procs= []
        self.fixed_one_side_alignment(pool, "SRC")
        self.fixed_one_side_alignment(pool, "TGT")
        # unpack the results
        pool.close()
        pool.join()
        self.unpack_procs("SRC")
        self.unpack_procs("TGT")
            
    def unpack_procs(self, flag="SRC"):
        _, _, procs, mappings = self.align_config(flag) 
        for p in procs:
            from_ind, from_entity_iri, to_entity_iri, mapping_value = p.get()
            if flag == "SRC":
                mappings.iloc[from_ind] = [from_entity_iri, to_entity_iri, mapping_value]
            else:
                mappings.iloc[from_ind] = [to_entity_iri, from_entity_iri, mapping_value]
            
    def align_config(self, flag="SRC"):
        assert flag == "SRC" or flag == "TGT"
        from_onto_lexicon = self.src_onto_lexicon if flag == "SRC" else self.tgt_onto_lexicon
        to_onto_lexicon = self.tgt_onto_lexicon if flag == "SRC" else self.src_onto_lexicon
        procs = self.src_procs if flag == "SRC" else self.tgt_procs
        mappings = self.src2tgt_mappings if flag == "SRC" else self.tgt2src_mappings
        return from_onto_lexicon, to_onto_lexicon, procs, mappings
            
    def fixed_one_side_alignment(self, pool, flag="SRC"):
        """Fix one ontology, generate the target entities for each entity in the fixed ontology"""
        from_onto_lexicon, to_onto_lexicon, procs, _ = self.align_config(flag) 
        
        for i, from_dp in from_onto_lexicon.iterrows():
            p = pool.apply_async(self.fix_one_entity_alignment, args=(i, from_dp, to_onto_lexicon, flag, ))
            procs.append(p)
    
    def fix_one_entity_alignment(self, from_ind, from_dp, to_onto_lexicon, flag="SRC"):
        from_entity_iri = from_dp["Entity-IRI"]
        from_entity_lexicon = Ontology.parse_entity_lexicon(from_dp["Entity-Lexicon"])[0]
        result = to_onto_lexicon["Entity-Lexicon"].apply(lambda to_entity_lexicon: 
            self.max_norm_edit_sim(from_entity_lexicon, Ontology.parse_entity_lexicon(to_entity_lexicon)[0]))
        to_entity_iri = to_onto_lexicon.iloc[result.idxmax()]["Entity-IRI"]
        mapping_value = result.max()
        self.log_print(f"[PID {os.getpid()}][{self.name}][{flag}: {self.src}][#Entity: {from_ind}][Mapping: {from_entity_iri}, {to_entity_iri}, {mapping_value}]" if flag == "SRC" \
            else f"[PID {os.getpid()}][{self.name}][{flag}: {self.tgt}][#Entity: {from_ind}][Mapping: {from_entity_iri}, {to_entity_iri}, {mapping_value}]")
        to_entity_iri = to_onto_lexicon.iloc[result.idxmax()]["Entity-IRI"]
        return from_ind, from_entity_iri, to_entity_iri, mapping_value

    @staticmethod    
    def max_norm_edit_sim(from_lexicon, to_lexicon):
        label_pairs = product(from_lexicon, to_lexicon)
        sim_scores = [levenshtein.normalized_similarity(src, tgt) for src, tgt in label_pairs]
        return max(sim_scores)
