"""
Pretrained BERT and its variants from Pytorch-based Huggingface Library.
"""
from transformers import AutoModel, AutoTokenizer
from typing import List
import torch


class PretrainedBERT:

    def __init__(self, pretrained_path):
        print("Load the Pretrained BERT model...")
        self.model = AutoModel.from_pretrained(pretrained_path, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.model.eval()

    def tokenize(self, sent: str):
        input_ids = self.tokenizer(sent)["input_ids"]
        return self.tokenizer.convert_ids_to_tokens(input_ids)
    
    def batch_tokenize(self, sents: List[str]):
        batch_input_ids = self.tokenizer(sents, padding=True, return_tensors="pt")["input_ids"]
        for input_ids in batch_input_ids:
            yield self.tokenizer.convert_ids_to_tokens(input_ids)
        

    def batch_word_embeds(self, sents: List[str], neg_layer_num=-2):
        """neg_layer_num: negative number of layer, e.g. -1 means the last layer,
           the default strategy is to take the embeddings from the secont-to-last (-2) layer
        """
        # dict with keys 'input_ids', 'token_type_ids' and 'attention_mask'
        inputs = self.tokenizer(sents, padding=True, return_tensors="pt")
        # dict with keys 'last_hidden_state', 'pooler_output' and 'hidden_states'
        mask = inputs["attention_mask"]  # (batch_size, max_sent_len)
        with torch.no_grad():
            outputs = self.model(**inputs)  # **inputs will give values whenever the keys are called
            batch_embeds = torch.stack(outputs["hidden_states"], dim=0)  # (#layer, batch_size, max_sent_len, hid_dim)
            # embeddings taken from an *exact* layer
            batch_embeds = batch_embeds[neg_layer_num]  # (batch_size, max_sent_len, hid_dim)
            batch_embeds, mask = torch.broadcast_tensors(batch_embeds, mask.unsqueeze(2))  # broadcast the mask tensor to (batch_size, max_sent_len, hid_dim)
            batch_embeds = torch.mul(batch_embeds, mask)  # the masked positions are zeros now
        
        return batch_embeds, mask

    def batch_sent_embeds_mean(self, sents: List[str], neg_layer_num=-1):
        """Take the mean word embedding of specified layer as the sentence embedding"""
        batch_word_embeds, mask = self.batch_word_embeds(sents, neg_layer_num=neg_layer_num)  # (batch_size, sent_len, hid_dim)
        sum_embeds = torch.sum(batch_word_embeds, dim=1)  # (batch_size, hid_dim)
        norm = torch.sum(mask , dim=1).double().pow(-1)  # (batch_size, hid_dim), storing the inverse of tokenized sentence length
        return torch.mul(sum_embeds, norm) 
    
    def batch_sent_embeds_cls(self, sents: List[str], neg_layer_num=-1):
        """Take the [cls] token embedding of specified layer as the sentence embedding"""
        batch_word_embeds, _ = self.batch_word_embeds(sents, neg_layer_num=neg_layer_num)  # (batch_size, sent_len, hid_dim)
        return batch_word_embeds[:, 0, :]  # (batch_size, hid_dim)
