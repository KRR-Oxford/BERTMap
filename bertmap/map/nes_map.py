"""
Direct Search Mapping Generation on using the *normalized edit distance score* as the distance metric.

Unlike the BERT experiment where the batched (vectorization) algorithm is used, here we apply the multiprocessing on each batch.
"""

from bertmap.onto import OntoBox
from bertmap.map import OntoMapping
from itertools import product
from textdistance import levenshtein
from multiprocessing_on_dill import Process
from typing import List, Optional
from owlready2.entity import ThingClass
import time
import os

class NormEditSimMapping(OntoMapping):
    
    def __init__(self, 
                 src_ob: OntoBox, 
                 tgt_ob: OntoBox,
                 candidate_limit: Optional[int] = 50,
                 save_dir: str=""):
        super().__init__(src_ob, tgt_ob, candidate_limit, save_dir)
            
    def alignment(self, flag: str="SRC") -> None:
        self.start_time = time.time()
        align_one = lambda x: self.align_one_class(x, flag=flag)
        from_ob, _ = self.from_to_config(flag=flag)
        for from_class in from_ob.onto.classes(): align_one(from_class)
    
    def align_one_class(self, from_class: ThingClass, flag: str) -> None:
        
        from_ob, to_ob = self.from_to_config(flag=flag)
        print_flag = f"{flag}: {self.src_ob.onto_text.iri_abbr}" if flag == "SRC" else f"{flag}: {self.tgt_ob.onto_text.iri_abbr}"

        from_class_iri = from_ob.onto_text.abbr_entity_iri(from_class.iri)
        from_labels = from_ob.onto_text.texts[from_class_iri]["label"]
        to_search_space = to_ob.onto_text.text.keys() if not self.candidate_limit \
            else to_ob.select_candidates(from_labels, self.candidate_limit)
        from_class_idx = from_ob.onto_text.class2idx[from_class_iri]
        if len(to_search_space) == 0:
            self.log_print(f"[Time: {round(time.time() - self.start_time)}][{print_flag}][Class-idx: {from_class_idx}] No candidates available for for current entity ...")
        
        max_sim_score= 0; max_sim_class = ""
        for to_class_iri in to_search_space:
            to_labels = to_ob.onto_text.texts[to_class_iri]["label"]
            sim_score = self.max_norm_edit_sim(from_labels, to_labels)
            if sim_score > max_sim_score:
                max_sim_score = sim_score
                max_sim_class = to_class_iri
                if max_sim_score == 1.0: break
        result = (from_class_iri, max_sim_class, max_sim_score)
        self.log_print(f"[Time: {round(time.time() - self.start_time)}][PID {os.getpid()}][{print_flag}][Class-idx: {from_class_idx}][Mapping: {result}]")

    @staticmethod    
    def max_norm_edit_sim(from_labels: List[str], to_labels: List[str]) -> float:
        label_pairs = product(from_labels, to_labels)
        sim_scores = [levenshtein.normalized_similarity(src, tgt) for src, tgt in label_pairs]
        return max(sim_scores)
