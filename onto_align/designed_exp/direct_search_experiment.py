"""Direct Search Experiment on using some kind of normalized distance metric:

   Algorithm:
   
        Compute the *Value between each source-target entity pair where Value is defined by:
           Dist = min(norm_dist(entity1.labels, entity2.labels))
           Value = 1 - Dist
           
        [Fix the source side] 
            For each source entity (entity1), pick the target entity (entity2) according to the min(Dist) or max(Value)
            
        [Fix the target side] 
            For each target entity (entity2), pick the source entity (entity1) according to the min(Dist) opr max(Value)
            
        Remove the duplicates
"""
from onto_align.onto import Ontology, OntoExperiment
import multiprocessing
import pandas as pd
import random
import os
import sys
import time


class DirectSearchExperiment(OntoExperiment):
    
    def __init__(self, src_onto_iri_abbr, tgt_onto_iri_abbr, 
                 src_data_tsv, tgt_data_tsv, save_path, 
                 task_suffix="small", exp_name="norm_edit_dist"):
        
        super().__init__(src_onto_iri_abbr, tgt_onto_iri_abbr, src_data_tsv, tgt_data_tsv, 
                         save_path, task_suffix=task_suffix, exp_name=exp_name)
        
        # result mappings: fixed src, search tgt; fixed tgt sea
        self.src2tgt_mappings_tsv = pd.DataFrame(index=range(len(self.src_tsv)), columns=["Entity1", "Entity2", "Value"])
        self.tgt2src_mappings_tsv = pd.DataFrame(index=range(len(self.tgt_tsv)), columns=["Entity1", "Entity2", "Value"])
        self.combined_mappings_tsv = None
    
    def run(self):
        t_start=time.time()
        src_batch_dict_list = []
        tgt_batch_dict_list = []
        
        # num_splits = num_intervals - 1 e.g. -|-|- 
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        num_splits = (multiprocessing.cpu_count() - 2) // 2  
        src_batch_iter = self.interval_split(num_splits, len(self.src_tsv))
        tgt_batch_iter = self.interval_split(num_splits, len(self.tgt_tsv))
        
        count = 0
        for src_batch_inds in src_batch_iter:
            count += 1
            src_batch_dict_list.append(pool.apply_async(self.batch_mappings, args=(src_batch_inds, False, )))
        assert count == num_splits + 1
        
        count = 0
        for tgt_batch_inds in tgt_batch_iter:
            count += 1
            tgt_batch_dict_list.append(pool.apply_async(self.batch_mappings, args=(tgt_batch_inds, True, )))
        assert count == num_splits + 1
        
        pool.close()
        pool.join()
        
        for result in src_batch_dict_list:
            for k, v in result.get().items():
                self.src2tgt_mappings_tsv.iloc[k] = v   
        for result in tgt_batch_dict_list:
            for k, v in result.get().items():
                self.tgt2src_mappings_tsv.iloc[k] = v   
        
        t_end = time.time()
        t = t_end-t_start
        self.log_print('the program time is :%s' %t)
    
    def batch_mappings(self, batch_inds, inverse=False):
        """Generate a batch of mappings for given source or target (inverse=True) entity batch indices"""
        
        flag = "Fixed-SRC" if not inverse else "Fixed-TGT"
        pid = os.getpid()
        size = len(batch_inds)
        self.log_print(f"[Process {pid}][{flag}] Starting a batch with size {size}")
        
        batch_dict = dict()
        src_tsv = self.src_tsv if not inverse else self.tgt_tsv
        tgt_tsv = self.tgt_tsv if not inverse else self.src_tsv
        
        for i in batch_inds:
            src_row = src_tsv.iloc[i]
            src_entity_iri = src_row["entity-iri"]
            src_lexicon = self.lexicon_process(src_row["entity-lexicon"])
            min_dist = 1
            tgt_entity_iri = None
            
            for j in range(len(tgt_tsv)):
                tgt_row = tgt_tsv.iloc[j]
                tgt_lexicon = self.lexicon_process(tgt_row["entity-lexicon"])
                entity_dist = self.entity_dist_metric(src_lexicon, tgt_lexicon)
                if (entity_dist < min_dist) or (entity_dist == min_dist and random.random() < 0.5):
                    min_dist = entity_dist
                    tgt_entity_iri = tgt_row["entity-iri"]   
                
                print(f"[Process {pid}][{flag}] current [{j}]: {entity_dist:5f}; stored: {min_dist}")
                sys.stdout.flush()
            
            if not inverse:
                batch_dict[i] = [Ontology.reformat_entity_uri(src_entity_iri, self.src_iri), 
                                 Ontology.reformat_entity_uri(tgt_entity_iri, self.tgt_iri), 
                                 1 - min_dist]
            else:
                batch_dict[i] = [Ontology.reformat_entity_uri(tgt_entity_iri, self.src_iri), 
                                 Ontology.reformat_entity_uri(src_entity_iri, self.tgt_iri), 
                                 1 - min_dist]      
                
            self.log_print(f"[Map {i}] {batch_dict[i][0]} | {batch_dict[i][1]}")         
        
        self.log_print(f"[Process {pid}][{flag}] Finishing the batch ...")
            
        return batch_dict
    
    def lexicon_process(self, entity_lexicon):
        raise NotImplementedError
    
    def entity_dist_metric(self, src_lexicon, tgt_lexicon):
        raise NotImplementedError
    
    def save(self):
        self.combined_mappings_tsv = self.src2tgt_mappings_tsv.append(self.tgt2src_mappings_tsv).drop_duplicates()
        name = f"{self.src}2{self.tgt}_{self.task_suffix}-{self.exp_name}"
        self.src2tgt_mappings_tsv.to_csv(f"{self.save_path}/{name}-src2tgt.tsv", index=False, sep='\t')
        self.tgt2src_mappings_tsv.to_csv(f"{self.save_path}/{name}-tgt2src.tsv", index=False, sep='\t')
        self.combined_mappings_tsv.to_csv(f"{self.save_path}/{name}-combined.tsv", index=False, sep='\t')