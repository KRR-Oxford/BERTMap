"""Direct Search Experiment on using some kind of normalized distance metric:

   Naive Algorithm:
   
        Compute the *Value between each source-target entity pair where Value is defined by:
           Dist = norm_distance(entity1, entity2)
           Value = norm_similarity(entity1, entity2)
           
        [Fix the source side] 
            For each source entity (entity1), pick the target entity (entity2) according to the min(Dist) or max(Value)
            
        [Fix the target side] 
            For each target entity (entity2), pick the source entity (entity1) according to the min(Dist) opr max(Value)
            
        Remove the duplicates
        
    Batched Algorithm with Divide and Conquer Design:
    
        [Fix the source side] 
            For each *source batch* of entities, compute the Value for the source batch against *each target batch* (divide),
            merge the output Value(s) (conquer) to finalize the mappings for current source batch.
            
        [Fix the target side]
            Repeat the similar step as above except that we inverse the source-target direction.
            
    Possible Improvement: we can compute batch-to-batch for source and target sides simultaneously. However, it won't be practical for very large ontologies.
"""
from ontoalign.onto import Ontology
from ontoalign.experiments import OntoAlignExperiment
from ontoalign.utils import swap
import pandas as pd



class DirectSearchExperiment(OntoAlignExperiment):
    
    def __init__(self, src_onto_iri_abbr, tgt_onto_iri_abbr, 
                 src_onto_lexicon_tsv, tgt_onto_lexicon_tsv, save_path, 
                 src_batch_size=10000, tgt_batch_size=10000,
                 task_suffix="small", name="direct_search_exp"):
        
        super().__init__(src_onto_iri_abbr, tgt_onto_iri_abbr, src_onto_lexicon_tsv, tgt_onto_lexicon_tsv, 
                         save_path, task_suffix=task_suffix, name=name)
        
        self.src_batch_size = src_batch_size
        self.tgt_batch_size = tgt_batch_size
    
    def alignment(self):
        self.src2tgt_mappings = self.fixed_one_side_alignment("SRC")
        self.tgt2src_mappings = self.fixed_one_side_alignment("TGT")
        self.combined_mappings = self.src2tgt_mappings.append(self.tgt2src_mappings).drop_duplicates()
            
    def fixed_one_side_alignment(self, flag="SRC"):
        """Fix one ontology, generate the target entities for each entity in the fixed ontology"""
        
        assert flag == "SRC" or flag == "TGT"
        from_onto_lexicon_path = self.src_onto_lexicon_path
        to_onto_lexicon_path = self.tgt_onto_lexicon_path
        from_batch_size = self.src_batch_size
        to_batch_size = self.tgt_batch_size
        mappings = pd.DataFrame(index=range(len(self.src_onto_lexicon)), columns=["Entity1", "Entity2", "Value"])
        if flag == "TGT":
            from_onto_lexicon_path, to_onto_lexicon_path = swap(from_onto_lexicon_path, to_onto_lexicon_path)
            from_batch_size, to_batch_size = swap(from_batch_size, to_batch_size)
            mappings = pd.DataFrame(index=range(len(self.tgt_onto_lexicon)), columns=["Entity1", "Entity2", "Value"])
        
        from_batch_generator = Ontology.iri_lexicon_batch_generator(from_onto_lexicon_path, batch_size=from_batch_size)
        to_batch_generator = Ontology.iri_lexicon_batch_generator(to_onto_lexicon_path, batch_size=to_batch_size)
        
        batch_ind = 0
        for batch in from_batch_generator:
            # to-generator needs to be re-init for every from-batch
            to_batch_generator = Ontology.iri_lexicon_batch_generator(to_onto_lexicon_path, batch_size=to_batch_size) 
            to_entity_iris, mapping_values = self.batch_alignment(batch, to_batch_generator, flag=flag)
            from_entity_iris = batch["Entity-IRI"]
            mappings["Entity1"].iloc[batch.index] = from_entity_iris
            mappings["Entity2"].iloc[batch.index] = to_entity_iris
            mappings["Value"].iloc[batch.index] = mapping_values
            self.log_print(f"[{flag}: {self.src}][Batch: {batch_ind}]" if flag == "SRC" else f"[{flag}: {self.tgt}][Batch: {batch_ind}]")
            batch_ind += 1
        
        # swap the mapping direction for target side for furhter combination
        if flag == "TGT":
            temp = mappings["Entity1"]
            mappings["Entity1"] = mappings["Entity2"]
            mappings["Entity2"] = temp
            
        return mappings

    def batch_alignment(self, from_batch, to_batch_generator, flag="SRC"):
        raise NotImplementedError
            
    def save(self):
        self.src2tgt_mappings.to_csv(f"{self.save_path}/{self.src}2{self.tgt}.{self.task_suffix}.{self.name}.tsv", index=False, sep='\t')
        self.tgt2src_mappings.to_csv(f"{self.save_path}/{self.tgt}2{self.src}.{self.task_suffix}.{self.name}.tsv", index=False, sep='\t')
        self.combined_mappings.to_csv(f"{self.save_path}/{self.src}-{self.tgt}-combined.{self.task_suffix}.{self.name}.tsv", index=False, sep='\t')