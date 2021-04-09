from bertmap.bert import PretrainedBERT
from bertmap.onto import Ontology
import torch
import itertools
from typing import List

class BERTClassEmbedding:
    
    def __init__(self, pretrained_bert: PretrainedBERT, neg_layer_num=-1):
        self.bert = pretrained_bert
        self.neg_layer_num = neg_layer_num
        
    def class_embeds_from_ontology(self, batch_sent_embeds_method: str, iri_lexicon_file, batch_size=1000):
        entity_embeds = []
        batch_ind = 0
        for batch in Ontology.class2text_batch_generator(iri_lexicon_file, batch_size=batch_size):
            batch_lexicon_sents = []
            batch_lexicon_sizes = []
            for lexicon in batch["Class-Text"]:
                parsed_lexicon_sents, lexicon_size = Ontology.parse_class_text(lexicon)
                batch_lexicon_sents.append(parsed_lexicon_sents)
                batch_lexicon_sizes.append(lexicon_size)
            batch_lexicon_sents = list(itertools.chain.from_iterable(batch_lexicon_sents))
            entity_embeds.append(self.class_embeds_from_batched_class2text(batch_sent_embeds_method, batch_lexicon_sizes, batch_lexicon_sents))
            print(f"[Batch {batch_ind}] Finish the class embeddings ...")
            batch_ind += 1
        return torch.cat(entity_embeds, dim=0)
        
    def class_embeds_from_batched_class2text(self, batch_sent_embeds_method: str, batch_class2text_sizes: List[int], batch_class2text_sents: List[str]):
        """Generate entity lexicon embedding for a batch of entities of different lexicon sizes;
           An entity may have *multiple* lexicon sentences, we take the mean of these lexicon embeddings as the entity embedding;
           To this end, we need to know the {number of lexicon sentences} in the batch for each entity.

        Args:
            batch_sent_embed_method: specify the embedding *function*
            batch_lexicon_sizes (List[int]): specify the lexicon size for each entity in the batch
            batch_lexicon (List[str]): ordered lexicon sentences for all entities in the batch
        """
        num_classes = len(batch_class2text_sizes)
        all_class2text_sents_embeds = getattr(self.bert, batch_sent_embeds_method)(batch_class2text_sents, neg_layer_num=self.neg_layer_num)  # (batch_(sents)_size, hid_dim)
        # return all_lexicon_sents_embeds
        class_embeds_list = []
        end = 0
        for i in range(len(batch_class2text_sizes)):
            start = end
            end = batch_class2text_sizes[i] + start
            # print(start, end)
            class_text_embeds = all_class2text_sents_embeds[start: end]
            class_embed = torch.mean(class_text_embeds, dim=0)  # here class embedding = mean(textual sents embedding)
            class_embeds_list.append(class_embed)
        assert len(class_embeds_list) == num_classes
        return torch.stack(class_embeds_list)
