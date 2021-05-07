"""
Direct Search Mapping Generation on using the *normalized edit distance score* as the distance metric.

Unlike the BERT experiment where the batched (vectorization) algorithm is used, here we apply the multiprocessing on each batch.
"""

from bertmap.onto import OntoBox
from bertmap.map import OntoMapping
from itertools import product
from textdistance import levenshtein
from multiprocessing_on_dill import Pool
from typing import List, Optional
import time
import os

class NormEditSimMapping(OntoMapping):
    
    def __init__(self, 
                 src_ob: OntoBox, 
                 tgt_ob: OntoBox,
                 candidate_limit: Optional[int] = 50,
                 num_pools: int=18):
        super().__init__(src_ob, tgt_ob, candidate_limit)
        self.num_pools = num_pools

    def run(self) -> None:
        self.pool = Pool(self.num_pools)
        t_start = time.time()
        self.alignment("SRC"); self.alignment("TGT")
        self.pool.close(); self.pool.join()
        t_end = time.time()
        self.log_print(f'the overall program time is :{t_end - t_start}')
            
    def alignment(self, flag: str="SRC") -> None:
        self.start_time = time.time()  # start time for one-side alignment
        from_ob, to_ob = self.from_to_config(flag=flag)
        for from_class_iri, text_dict in from_ob.onto_text.texts.items():
            labels = text_dict["label"]
            search_space = to_ob.onto_text.text.keys() if not self.candidate_limit \
                else to_ob.select_candidates(labels, self.candidate_limit)
            self.pool.apply_async(self.align_one_class, args=(from_class_iri, search_space, flag, ))
    
    def align_one_class(self, 
                        from_class_iri: str, 
                        to_search_space: List[str], 
                        to_ob: OntoBox,
                        flag: str) -> None:
        from_ob, to_ob = self.from_to_config(flag=flag)
        from_class_idx = from_ob.onto_text.class2idx[from_class_iri]
        print_flag = f"{flag}: {self.src}" if flag == "SRC" else f"{flag}: {self.tgt}"
        if len(to_search_space) == 0:
            self.log_print(f"[Time: {round(time.time() - self.start_time)}][{self.name}][{print_flag}]\
                [#Class: {from_class_idx}] No candidates available for for current entity ..."); return
        max_sim_score= 0; max_sim_class = ""
        from_labels = from_ob.onto_text.texts["label"]
        for to_class_iri in to_search_space:
            to_labels = to_ob.onto_text.texts[to_class_iri]["label"]
            sim_score = self.max_norm_edit_sim(from_labels, to_labels)
            if sim_score > max_sim_score:
                max_sim_score = sim_score
                max_sim_class = to_class_iri
                if max_sim_score == 1.0: break
        result = (from_class_iri, max_sim_class, max_sim_score)
        self.log_print(f"[Time: {round(time.time() - self.start_time)}][PID {os.getpid()}]\
            [{self.name}][{print_flag}][#Class: {from_class_idx}][Mapping: {result}]")

    @staticmethod    
    def max_norm_edit_sim(from_labels: List[str], to_labels: List[str]) -> float:
        label_pairs = product(from_labels, to_labels)
        sim_scores = [levenshtein.normalized_similarity(src, tgt) for src, tgt in label_pairs]
        return max(sim_scores)
