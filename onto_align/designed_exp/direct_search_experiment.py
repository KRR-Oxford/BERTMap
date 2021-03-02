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
from onto_align.onto import OntoExperiment
import multiprocessing
import pandas as pd


class DirectSearchExperiment(OntoExperiment):
    
    def __init__(self, src_onto_iri_abbr, tgt_onto_iri_abbr, src_data_tsv, tgt_data_tsv, task="small", exp_name="norm_edit_dist"):
        super().__init__(src_onto_iri_abbr, tgt_onto_iri_abbr, src_data_tsv, tgt_data_tsv, task=task, exp_name=exp_name)
        
        # result mappings: fixed src, search tgt; fixed tgt sea
        self.src2tgt_mappings_tsv = pd.DataFrame(index=range(len(self.src_tsv)), columns=["Entity1", "Entity2", "Value"])
        self.tgt2src_mappings_tsv = pd.DataFrame(index=range(len(self.tgt_tsv)), columns=["Entity1", "Entity2", "Value"])
        self.combined_mappings_tsv = None
    
    def run(self):
        src_queue = multiprocessing.SimpleQueue()
        tgt_queue = multiprocessing.SimpleQueue()
        # -2 cores to prevent overflow
        # num_splits = num_intervals - 1 e.g. -|-|- 
        num_splits = (multiprocessing.cpu_count() - 2) // 2  
        src_batch_iter = self.interval_split(num_splits, len(self.src_tsv))
        tgt_batch_iter = self.interval_split(num_splits, len(self.tgt_tsv))
        count = 0
        for src_batch_inds in src_batch_iter:
            count += 1
            print(f"Compute the mapping for {str(count)}th source entity batch ...")
            p = multiprocessing.Process(target=self.batch_mappings, args=(src_batch_inds, src_queue, False, )) 
            p.start()

        assert count == num_splits + 1
        count = 0
        for tgt_batch_inds in tgt_batch_iter:
            count += 1
            print(f"Compute the mapping for {str(count)}th target entity batch ...")
            p = multiprocessing.Process(target=self.batch_mappings, args=(tgt_batch_inds, tgt_queue, True, )) 
            p.start() 
        assert count == num_splits + 1
        
        for _ in range(count):
            for k, v in src_queue.get().items():
                self.src2tgt_mappings_tsv.iloc[k] = v   
            for k, v in tgt_queue.get().items():
                self.tgt2src_mappings_tsv.iloc[k] = v   
        assert src_queue.empty() and tgt_queue.empty()
    
    def batch_mappings(self, batch_inds, queue=None, inverse=False):
        raise NotImplementedError
    
    def save(self, save_path):
        self.combined_mappings_tsv = self.src2tgt_mappings_tsv.append(self.tgt2src_mappings_tsv).drop_duplicates()
        self.src2tgt_mappings_tsv.to_csv(f"{save_path}/{self.src}2{self.tgt}_{self.task}-{self.exp_name}-src2tgt.tsv", index=False, sep='\t')
        self.tgt2src_mappings_tsv.to_csv(f"{save_path}/{self.src}2{self.tgt}_{self.task}-{self.exp_name}-tgt2src.tsv", index=False, sep='\t')
        self.combined_mappings_tsv.to_csv(f"{save_path}/{self.src}2{self.tgt}_{self.task}-{self.exp_name}-combined.tsv", index=False, sep='\t')