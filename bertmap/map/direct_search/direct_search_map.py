"""
Direct Search Mapping Generation superclass on using some kind of normalized distance metric or classifier (from fine-tuned BERT):

   Prelimniary Algorithm (One-side-fixed Search):
   
        Compute the *Value between each source-target entity pair where Value is defined by:
           Dist = norm_distance(entity1, entity2)
           Value = norm_similarity(entity1, entity2)
           
        [Fix the source side] 
            For each source entity (entity1), pick the target entity (entity2) according to the min(Dist) or max(Value)
            
        [Fix the target side] 
            For each target entity (entity2), pick the source entity (entity1) according to the min(Dist) opr max(Value)
            
        Remove the duplicates
"""

from bertmap.map import OntoMapping
import pandas as pd


class DirectSearchMapping(OntoMapping):
    
    def __init__(self, src_onto_iri_abbr, tgt_onto_iri_abbr, 
                 src_onto_class2text_tsv, tgt_onto_class2text_tsv, save_path, 
                 task_suffix="small", name="direct_search_exp"):
        super().__init__(src_onto_iri_abbr, tgt_onto_iri_abbr, src_onto_class2text_tsv, tgt_onto_class2text_tsv, 
                    save_path, task_suffix=task_suffix, name=name)
        
        self.src2tgt_mappings = pd.DataFrame(index=range(len(self.src_onto_class2text)), columns=["Entity1", "Entity2", "Value"])
        self.tgt2src_mappings = pd.DataFrame(index=range(len(self.tgt_onto_class2text)), columns=["Entity1", "Entity2", "Value"])
        self.combined_mappings = None
        
    def alignment(self):
        # fix SRC side
        self.fixed_one_side_alignment("SRC")
        self.src2tgt_mappings.to_csv(f"{self.save_path}/src2tgt.{self.src}2{self.tgt}.{self.task_suffix}.{self.name}.tsv", index=False, sep='\t')
        # fix TGT side
        self.fixed_one_side_alignment("TGT")
        self.tgt2src_mappings.to_csv(f"{self.save_path}/tgt2src.{self.src}2{self.tgt}.{self.task_suffix}.{self.name}.tsv", index=False, sep='\t')
        # generate combined mappings
        self.combined_mappings = self.src2tgt_mappings.append(self.tgt2src_mappings).drop_duplicates().dropna()
        self.combined_mappings.to_csv(f"{self.save_path}/combined.{self.src}2{self.tgt}.{self.task_suffix}.{self.name}.tsv", index=False, sep='\t')
        
    def align_config(self, flag="SRC"):
        """Configurations for swithcing the fixed ontology side."""
        raise NotImplementedError
    
    def fixed_one_side_alignment(self, flag="SRC"):
        raise NotImplementedError
