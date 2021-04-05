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
        self.fixed_one_side_alignment("SRC")
        self.fixed_one_side_alignment("TGT")
        
    def align_config(self, flag="SRC"):
        """Configurations for swithcing the fixed ontology side."""
        raise NotImplementedError
    
    def fixed_one_side_alignment(self, flag="SRC"):
        raise NotImplementedError
        
    def save(self):
        self.combined_mappings = self.src2tgt_mappings.append(self.tgt2src_mappings).drop_duplicates().dropna()
        self.src2tgt_mappings.to_csv(f"{self.save_path}/{self.src}2{self.tgt}.{self.task_suffix}.{self.name}.tsv", index=False, sep='\t')
        self.tgt2src_mappings.to_csv(f"{self.save_path}/{self.tgt}2{self.src}.{self.task_suffix}.{self.name}.tsv", index=False, sep='\t')
        self.combined_mappings.to_csv(f"{self.save_path}/{self.src}-{self.tgt}-combined.{self.task_suffix}.{self.name}.tsv", index=False, sep='\t')