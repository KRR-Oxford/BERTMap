"""Mapping Generation on using Pretrained/Fine-tuned BERT with various pooling strategies and cosine-similarity:

    Batched Algorithm with Divide and Conquer Design:
    
        [Fix the source side] 
            For each *source batch* of classes, compute the Value for the source batch against *each target batch* (divide),
            merge the output Value(s) (conquer) to finalize the mappings for current source batch.
            
        [Fix the target side]
            Repeat the similar step as above except that we inverse the source-target direction.
            
    Possible Improvement: we can compute batch-to-batch for source and target sides simultaneously. However, it won't be practical for very large ontologies.
"""

from bertmap.onto import OntoBox
from bertmap.map import OntoMapping
from bertmap.bert import BERTStatic
from bertmap.utils import get_device
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional
import torch
import time
import pandas as pd


class BERTEmbedsMapping(OntoMapping):
    
    def __init__(self,
                 src_ob: OntoBox,
                 tgt_ob: OntoBox,
                 candidate_limit: Optional[int] = 50,
                 batch_size: int=32,
                 nbest: int=1, 
                 bert_checkpoint: str="some checkpoint", 
                 tokenizer_path: str="emilyalsentzer/Bio_ClinicalBERT", 
                 string_match: bool=True, 
                 strategy: str="mean",
                 device_num: int=0):
        
        super().__init__(src_ob, tgt_ob, candidate_limit)
        # basic attributes 
        self.batch_size = batch_size   
        self.nbest = nbest
        self.string_match = string_match
        self.strategy = strategy
        assert self.strategy == "mean" or self.strategy == "max"
        
        self.bert = BERTStatic(bert_checkpoint=bert_checkpoint, with_classifier=False)
        self.device = get_device(device_num=device_num)
        self.bert.model.to(self.device)
  
    def fixed_one_side_alignment(self, flag="SRC"):
        """Fix one ontology, generate the target entities for each entity in the fixed ontology"""
        # configurations
        self.start_time = time.time()     
        from_onto_class2text_path, to_onto_class2text_path, _, to_index, map_name = self.align_config(flag)
        from_onto_class2text = OntoBox.load_classtexts(from_onto_class2text_path)
        to_onto_class2text = OntoBox.load_classtexts(to_onto_class2text_path)
        results = []
        for i, dp in from_onto_class2text.iterrows():
            from_labels, from_len = OntoBox.parse_class_text(dp["Class-Text"])
            search_space = to_onto_class2text if not to_index else self.select_candidates(dp["Class-Text"], flag=flag)
            if len(search_space) == 0:
                self.log_print("[Time: {round(time.time() - self.start_time)}][{self.name}][{print_flag}][#Class: {i}] No candidates available for for current entity ...")
                continue
            to_batch_generator = OntoBox.batch_iterator(search_space, batch_size=self.batch_size)
            nbest_results = self.batch_alignment(from_labels, from_len, to_batch_generator, self.batch_size, flag=flag)
            # collect the results
            for to_class_ind, mapping_score in nbest_results:
                if mapping_score <= 0.01:
                    mapping_score = 0.0
                to_class_iri = search_space.iloc[to_class_ind]["Class-IRI"]
                result = (dp["Class-IRI"], to_class_iri, mapping_score) if flag == "SRC" else (to_class_iri, dp["Class-IRI"], mapping_score)
                results.append(result)
                print_flag = f"{flag}: {self.src_ob.onto_text.iri_abbr}" if flag == "SRC" else f"{flag}: {self.tgt_ob.onto_text.iri_abbr}"
                self.log_print(f"[Time: {round(time.time() - self.start_time)}][{self.name}][{print_flag}][#Class: {i}][Mapping: {result}]")
        setattr(self, map_name, pd.DataFrame(results, columns=["Entity1", "Entity2", "Value"]))

    def batch_alignment(self, from_labels, from_len, to_batch_generator, to_batch_size, flag="SRC"):
        batch_nbest_scores = torch.tensor([-1] * self.nbest).to(self.device)
        batch_nbest_indices = torch.tensor([-1] * self.nbest).to(self.device)
        # print(from_len, from_labels)
        from_embed = self.embed_func(from_len, from_labels)
        j = 0
        for to_batch in to_batch_generator:
            if self.string_match:
                for m, to_class_dp in to_batch.iterrows():
                    to_labels, _ = OntoBox.parse_class_text(to_class_dp["Class-Text"])
                    label_pairs = [[from_label, to_label] for to_label in to_labels for from_label in from_labels]
                    # return the map if the to-class has a label that is exactly the same as one of the labels of the from-class
                    for pair in label_pairs:
                        if pair[0] == pair[1]:
                            return [(m, 1.0)]
            to_batch_embeds = self.embed_func_batch(to_batch)
            # compare the cosine similarity scores between two batches
            sim_scores = torch.tensor(cosine_similarity(from_embed, to_batch_embeds)).to(self.device).squeeze(0)
            K = len(sim_scores) if len(sim_scores) < self.nbest else self.nbest
            nbest_scores, nbest_indices = torch.topk(sim_scores, k=K)
            nbest_indices += j * to_batch_size
            # we do the substituion for every batch to prevent from memory overflow
            batch_nbest_scores, temp_indices = torch.topk(torch.cat([batch_nbest_scores, nbest_scores]), k=self.nbest)
            batch_nbest_indices = torch.cat([batch_nbest_indices, nbest_indices])[temp_indices]
            j += 1
            
        return list(zip(batch_nbest_indices.cpu().detach().numpy(), batch_nbest_scores.cpu().detach().numpy()))
