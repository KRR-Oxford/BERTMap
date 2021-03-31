"""Direct Search BERT Experiment on using some kind of normalized distance metric:

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

from bertmap.onto import Ontology
from bertmap.map.direct_search import DirectSearchMapping
from bertmap.embed import PretrainedBERT
from sklearn.metrics.pairwise import cosine_similarity
import torch

class DirectBERTMapping(DirectSearchMapping):
    
    def __init__(self, src_onto_iri_abbr, tgt_onto_iri_abbr, 
                 src_onto_lexicon_tsv, tgt_onto_lexicon_tsv, save_path, 
                 src_embeds_pt, tgt_embeds_pt,
                 src_batch_size=10000, tgt_batch_size=10000,
                 task_suffix="small", name="bc-mean", 
                 bert_path="emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__(src_onto_iri_abbr, tgt_onto_iri_abbr, src_onto_lexicon_tsv, tgt_onto_lexicon_tsv, save_path, 
                         task_suffix=task_suffix, name=name)
        self.src_batch_size = src_batch_size
        self.tgt_batch_size = tgt_batch_size
        self.bert = PretrainedBERT(bert_path)
        self.src_embeds = torch.load(src_embeds_pt)
        self.tgt_embeds = torch.load(tgt_embeds_pt)
        
    def align_config(self, flag="SRC"):
        assert flag == "SRC" or flag == "TGT"
        from_onto_lexicon_path = self.src_onto_lexicon_path
        to_onto_lexicon_path = self.tgt_onto_lexicon_path
        from_batch_size = self.src_batch_size
        to_batch_size = self.tgt_batch_size
        mappings = self.src2tgt_mappings
        if flag == "TGT":
            from_onto_lexicon_path, to_onto_lexicon_path = to_onto_lexicon_path, from_onto_lexicon_path
            from_batch_size, to_batch_size = to_batch_size, from_batch_size
            mappings = self.tgt2src_mappings
        return from_onto_lexicon_path, from_batch_size, to_onto_lexicon_path, to_batch_size, mappings

  
    def fixed_one_side_alignment(self, flag="SRC"):
        """Fix one ontology, generate the target entities for each entity in the fixed ontology"""
        
        from_onto_lexicon_path, from_batch_size, to_onto_lexicon_path, to_batch_size, mappings = self.align_config(flag)
        from_batch_generator = Ontology.iri_lexicon_batch_generator(from_onto_lexicon_path, batch_size=from_batch_size)
        batch_ind = 0
        for batch in from_batch_generator:
            # to-generator needs to be re-init for every from-batch
            to_batch_generator = Ontology.iri_lexicon_batch_generator(to_onto_lexicon_path, batch_size=to_batch_size) 
            to_entity_iris, mapping_values = self.batch_alignment(batch, to_batch_generator, flag=flag)
            from_entity_iris = batch["Entity-IRI"]
            mappings["Entity1"].iloc[batch.index] = from_entity_iris
            mappings["Entity2"].iloc[batch.index] = to_entity_iris
            mappings["Value"].iloc[batch.index] = mapping_values
            self.log_print(f"[{self.name}][{flag}: {self.src}][Batch: {batch_ind}] finished." if flag == "SRC" \
                else f"[{self.name}][{flag}: {self.tgt}][Batch: {batch_ind}] finished.")
            batch_ind += 1
        
        # swap the mapping direction for target side for furhter combination
        if flag == "TGT":
            mappings["Entity1"], mappings["Entity2"] = list(mappings["Entity2"]), list(mappings["Entity1"])


    def batch_alignment(self, from_batch, to_batch_generator, flag="SRC"):
        assert flag == "SRC" or flag == "TGT"
        from_batch_embeds = self.src_embeds[from_batch.index] if flag == "SRC" else self.tgt_embeds[from_batch.index]
        to_embeds = self.tgt_embeds if flag == "SRC" else self.src_embeds
        to_onto_lexicon = self.tgt_onto_lexicon if flag == "SRC" else self.src_onto_lexicon
        
        max_scores_list = []
        argmax_scores_list = []
        j = 0
        for to_batch in to_batch_generator:
            to_batch_embeds = to_embeds[to_batch.index]
            # compare the cosine similarity scores between two batches
            sim_scores = cosine_similarity(from_batch_embeds, to_batch_embeds)
            # pick the maximum/argmax scores in the to_batch w.r.t from_batch 
            # we need to add j * len(to_batch) to ensure the correct to-entity indices
            max_scores, argmax_scores = sim_scores.max(axis=1), sim_scores.argmax(axis=1) + j * len(to_batch)
            max_scores_list.append(torch.tensor(max_scores))
            argmax_scores_list.append(torch.tensor(argmax_scores))
            j += 1
        # stack the max/argmax scores for all to_batches w.r.t from_batch
        batch_max_scores_all = torch.stack(max_scores_list)  # (len(to_batch_generator), len(from_batch))
        batch_argmax_scores_all = torch.stack(argmax_scores_list)
        # apply the max/argmax function again on the maximum scores from to_batches (note that the torch.tensor.max() provides both max and argmax)
        batch_max_scores, batch_argmax_scores_inds = batch_max_scores_all.max(axis=0)  # value, indices
        # select the to-entity indices according the final argmax scores
        batch_argmax_scores = torch.gather(batch_argmax_scores_all, dim=0, index=batch_argmax_scores_inds.unsqueeze(0).repeat(j, 1))[0]
        return list(to_onto_lexicon["Entity-IRI"].iloc[list(batch_argmax_scores.numpy())]), batch_max_scores
