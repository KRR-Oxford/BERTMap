"""
Direct Search Mapping Generation on using the *normalized edit distance score* as the distance metric.

Unlike the BERT experiment where the batched (vectorization) algorithm is used, here we apply the multiprocessing on each batch.
"""

from bertmap.onto import Ontology
from bertmap.map.direct_search import DirectSearchMapping
from itertools import product
from textdistance import levenshtein
from multiprocessing_on_dill import Pool
import os

class DirectNESMapping(DirectSearchMapping):
    
    def __init__(self, src_onto_iri_abbr, tgt_onto_iri_abbr, 
                 src_onto_class2text_tsv, tgt_onto_class2text_tsv, save_path, 
                 task_suffix="small", num_pools=18):
        super().__init__(src_onto_iri_abbr, tgt_onto_iri_abbr, src_onto_class2text_tsv, tgt_onto_class2text_tsv, save_path, 
                         task_suffix=task_suffix, name="nes")
        
        self.num_pools = num_pools

    def alignment(self):
        pool = Pool(self.num_pools)
        src_procs = []
        tgt_procs= []
        self.fixed_one_side_alignment(pool, src_procs, "SRC")
        self.fixed_one_side_alignment(pool, tgt_procs, "TGT")
        # unpack the results
        pool.close()
        pool.join()
        for proc in src_procs:
            i, from_class_iri, to_class_iri, mapping_value = proc.get()
            self.src2tgt_mappings.iloc[i] = [from_class_iri, to_class_iri, mapping_value]
        for proc in tgt_procs:
            j, from_class_iri, to_class_iri, mapping_value = proc.get()
            self.tgt2src_mappings.iloc[j] = [to_class_iri, from_class_iri, mapping_value]
            
    def align_config(self, flag="SRC"):
        assert flag == "SRC" or flag == "TGT"
        from_onto_class2text = self.src_onto_class2text if flag == "SRC" else self.tgt_onto_class2text
        to_onto_class2text = self.tgt_onto_class2text if flag == "SRC" else self.src_onto_class2text
        return from_onto_class2text, to_onto_class2text
            
    def fixed_one_side_alignment(self, pool, procs, flag="SRC"):
        """Fix one ontology, generate the target entities for each entity in the fixed ontology"""
        from_onto_class2text, to_onto_class2text = self.align_config(flag) 
        
        for i, from_dp in from_onto_class2text.iterrows():
            p = pool.apply_async(self.fix_one_class_alignment, args=(i, from_dp, to_onto_class2text, flag, ))
            procs.append(p)
    
    def fix_one_class_alignment(self, from_ind, from_dp, to_onto_class2text, flag="SRC"):
        from_class_iri = from_dp["Class-IRI"]
        from_class_text = Ontology.parse_class_text(from_dp["Class-Text"])[0]
        max_sim_score= 0
        max_sim_ind = 0
        for j, dp in to_onto_class2text.iterrows():
            to_class_text = Ontology.parse_class_text(dp["Class-Text"])[0]
            sim_score = self.max_norm_edit_sim(from_class_text, to_class_text)
            if sim_score > max_sim_score:
                max_sim_score = sim_score
                max_sim_ind = j
                if max_sim_score == 1.0:
                    break
        to_class_iri = to_onto_class2text.iloc[max_sim_ind]["Class-IRI"]
        mapping_value = max_sim_score
        self.log_print(f"[PID {os.getpid()}][{self.name}][{flag}: {self.src}][#Entity: {from_ind}][Mapping: {from_class_iri}, {to_class_iri}, {mapping_value}]" if flag == "SRC" \
            else f"[PID {os.getpid()}][{self.name}][{flag}: {self.tgt}][#Entity: {from_ind}][Mapping: {from_class_iri}, {to_class_iri}, {mapping_value}]")
        
        return from_ind, from_class_iri, to_class_iri, mapping_value

    @staticmethod    
    def max_norm_edit_sim(from_class_text, to_class_text):
        label_pairs = product(from_class_text, to_class_text)
        sim_scores = [levenshtein.normalized_similarity(src, tgt) for src, tgt in label_pairs]
        return max(sim_scores)
