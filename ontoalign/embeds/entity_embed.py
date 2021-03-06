from ontoalign.embeds import PretrainedBert
from ontoalign.onto import Ontology
import torch
import itertools
from typing import List

class BertEntityEmbedding:
    
    def __init__(self, pretrained_bert: PretrainedBert):
        self.bert = pretrained_bert
        
    def entity_embeds_from_ontology(self, batch_sent_embeds_method: str, iri_lexicon_file, batch_size=1000):
        entity_embeds = []
        batch_ind = 0
        for batch in Ontology.iri_lexicon_batch_generator(iri_lexicon_file, batch_size=batch_size):
            batch_lexicon_sents = []
            batch_lexicon_sizes = []
            for lexicon in batch["Entity-Lexicon"]:
                parsed_lexicon_sents, lexicon_size = Ontology.parse_entity_lexicon(lexicon)
                batch_lexicon_sents.append(parsed_lexicon_sents)
                batch_lexicon_sizes.append(lexicon_size)
            batch_lexicon_sents = list(itertools.chain.from_iterable(batch_lexicon_sents))
            entity_embeds.append(self.entity_embeds_from_batch_lexicon(batch_sent_embeds_method, batch_lexicon_sizes, batch_lexicon_sents))
            print(f"[Batch {batch_ind}] Finish the entity embeddings ...")
            batch_ind += 1
        return torch.cat(entity_embeds, dim=0)
        
    def entity_embeds_from_batch_lexicon(self, batch_sent_embeds_method: str, batch_lexicon_sizes: List[int], batch_lexicon_sents: List[str]):
        """Generate entity lexicon embedding for a batch of entities of different lexicon sizes;
           An entity may have *multiple* lexicon sentences, we take the mean of these lexicon embeddings as the entity embedding;
           To this end, we need to know the {number of lexicon sentences} in the batch for each entity.

        Args:
            batch_sent_embed_method: specify the embedding *function*
            batch_lexicon_sizes (List[int]): specify the lexicon size for each entity in the batch
            batch_lexicon (List[str]): ordered lexicon sentences for all entities in the batch
        """
        num_entity = len(batch_lexicon_sizes)
        all_lexicon_sents_embeds = getattr(self.bert, batch_sent_embeds_method)(batch_lexicon_sents)  # (batch_(sents)_size, hid_dim)
        # return all_lexicon_sents_embeds
        entity_embeds_list = []
        end = 0
        for i in range(len(batch_lexicon_sizes)):
            start = end
            end = batch_lexicon_sizes[i] + start
            # print(start, end)
            entity_lexicon_embeds = all_lexicon_sents_embeds[start: end]
            entity_embed = torch.mean(entity_lexicon_embeds, dim=0)
            entity_embeds_list.append(entity_embed)
        assert len(entity_embeds_list) == num_entity
        return torch.stack(entity_embeds_list)
