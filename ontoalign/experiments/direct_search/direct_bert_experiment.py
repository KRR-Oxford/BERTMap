from ontoalign.experiments.direct_search import DirectSearchExperiment
from ontoalign.embeds import PretrainedBert
from sklearn.metrics.pairwise import cosine_similarity
import torch

class DirectBertExperiment(DirectSearchExperiment):
    
    def __init__(self, src_onto_iri_abbr, tgt_onto_iri_abbr, 
                 src_onto_lexicon_tsv, tgt_onto_lexicon_tsv, save_path, 
                 src_embeds_pt, tgt_embeds_pt,
                 src_batch_size=10000, tgt_batch_size=10000,
                 task_suffix="small", name="bc-mean", 
                 bert_path="emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__(src_onto_iri_abbr, tgt_onto_iri_abbr, src_onto_lexicon_tsv, tgt_onto_lexicon_tsv, save_path, 
                         src_batch_size=src_batch_size, tgt_batch_size=tgt_batch_size, task_suffix=task_suffix, name=name)
        self.bert = PretrainedBert(bert_path)
        self.src_embeds = torch.load(src_embeds_pt)
        self.tgt_embeds = torch.load(tgt_embeds_pt)
        

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
