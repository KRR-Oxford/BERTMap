"""
Pretrained BERT and its variants from Pytorch-based Huggingface Library.
"""
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine as cos_dist
from onto_align.onto import OntoExperiment
import torch
import itertools
import os
import sys


class PretrainedBert:

    def __init__(self, pretrained_path, embeds_save_path=""):
        print("Load the Pretrained BERT model...")
        self.model = AutoModel.from_pretrained(pretrained_path, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.model.eval()

    def tokenize(self, sent):
        input_ids = self.tokenizer(sent)["input_ids"]
        return self.tokenizer.convert_ids_to_tokens(input_ids)

    def get_word_embeddings(self, sent, neg_layer, strategy="exact"):
        """neg_layer: negative number of layer, e.g. -1 means the last layer"""
        # dict with keys 'input_ids', 'token_type_ids' and 'attention_mask'
        inputs = self.tokenizer(sent, return_tensors="pt")
        # dict with keys 'last_hidden_state', 'pooler_output' and 'hidden_states'
        with torch.no_grad():
            outputs = self.model(**inputs)  # **inputs will give values whenever the keys are called
            token_embeddings = torch.stack(outputs["hidden_states"], dim=0)  # (#layer, batch=1, sent_len, hid_dim)
            token_embeddings = torch.squeeze(token_embeddings, dim=1)   # (#layer, sent_len, hid_dim)
            if strategy == "exact":
                # embeddings taken from an *exact* layer
                # (sent_len, hid_dim)
                return token_embeddings[neg_layer, :, :]
            elif strategy == "sum":
                # embeddings taken from *sum* of the *last few* layers
                # (sent_len, hid_dim)
                return torch.sum(token_embeddings[neg_layer:], dim=0)
            elif strategy == "cat":
                # embeddings taken *from concatenation* of the *last few* layers
                sent_len = token_embeddings.size()[1]
                # (sent_len, hid_dim * #selected_layer)
                return token_embeddings[neg_layer:].permute(1, 0, 2).reshape(sent_len, -1)
            else:
                raise NameError("Pooling strategy not recognized! Available choices are: {exact}, {sum}, {cat}.")
            
    def get_basic_sent_embeddings(self, sent, strategy="last-2-mean"):
        """Provide basic sentence embedding by:
           1. taking the mean vector of the second-to-last layer (last-2-mean);
           2. taking the [cls] token of the last layer (last-1-cls)
        """
        if strategy == "last-2-mean":
            word_embeds = self.get_word_embeddings(sent, -2, strategy="exact")
            return torch.mean(word_embeds, dim=0)
        elif strategy == "last-1-cls":
            word_embeds = self.get_word_embeddings(sent, -1, strategy="exact")
            return word_embeds[0]
        else:
            raise NameError("Sentence embedding strategy not recognized! Available choices are: {last-2-mean}, {last-1-cls}.")
        
    def mean_lexicon_embeddings(self, iri2lexicon_file, embeds_save_path, strategy="last-2-mean", flag="Fixed-SRC"):
        # strategy "last-2-mean" or "last-1-cls"
        lexicon_embeds = []
        iri2lexicon_df = OntoExperiment.read_iri2lexicon_file(iri2lexicon_file)
        pid = os.getpid()
        for i, dp in iri2lexicon_df.iterrows():
            entity_lexicons = list(itertools.chain.from_iterable([property.split(" <sep> ") for property in dp["entity-lexicon"].split(" <property> ")]))
            tokenized_lexicon = [self.tokenizer.tokenize(lexicon) for lexicon in entity_lexicons]
            self.log_print(embeds_save_path, f"[Process {pid}][{flag}][{i}][{strategy}] {tokenized_lexicon}")
            stacked_embeds = torch.stack([self.get_basic_sent_embeddings(lexicon, strategy=strategy) for lexicon in entity_lexicons])
            lexicon_embeds.append(torch.mean(stacked_embeds, dim=0))
        
        lexicon_embeds = torch.stack(lexicon_embeds, dim=0)
        return lexicon_embeds
    
    @staticmethod
    def log_print(embeds_save_path, statement):
        print(statement)
        with open(f"{embeds_save_path}/embeds_generation.log", 'a+') as f:
            f.write(f'{statement}\n')
        sys.stdout.flush()
        

if __name__ == "__main__":
    # Similarity example
    text = "After stealing money from the bank vault, " \
           "the bank robber was seen fishing on the Mississippi river bank."
    bert = PretrainedBert('emilyalsentzer/Bio_ClinicalBERT')
    print("Tokenized text:", bert.tokenize(text))
    last_2_exact = bert.get_word_embeddings(text, -2, "exact")
    last_1_exact = bert.get_word_embeddings(text, -1, "exact")
    last_4_sum = bert.get_word_embeddings(text, -4, "sum")
    last_2_sum = bert.get_word_embeddings(text, -2, "sum")
    last_4_cat = bert.get_word_embeddings(text, -4, "cat")

    embed_dict = {"last-2-exact": last_2_exact,
                  "last-1-exact": last_1_exact,
                  "last-4-sum": last_4_sum,
                  "last-2-sum": last_2_sum,
                  "last-4-cat": last_4_cat}
    for strat, embeds in embed_dict.items():
        print(f"Pooling Strategy: {strat}.")
        # cosine here means the cosine distance
        diff_bank = 1 - cos_dist(embeds[10], embeds[19])  # it's equal to cos(a, b)
        same_bank = 1 - cos_dist(embeds[10], embeds[6])
        print('First 5 vector values for each instance of "bank".')
        print("[bank] vault:\t", str(embeds[6][:5]))   # "bank" at position 6 is contextually closer to at 10
        print("[bank] robber:\t", str(embeds[10][:5]))  # and contextually far apart from at 19
        print("river [bank]:\t", str(embeds[19][:5]))
        print('Vector similarity for  *similar*  meanings:  %.2f' % same_bank)
        print('Vector similarity for *different* meanings:  %.2f' % diff_bank)
        print("------------------")
