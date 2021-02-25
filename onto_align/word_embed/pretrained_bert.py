"""
Pretrained BERT and its variants from Pytorch-based Huggingface Library.
"""
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine
import torch


class PretrainedBert:

    def __init__(self, pretrained_path):
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
                raise NameError("Pooling strategy not recognized! Available choices are: {sum}, {cat}.")


if __name__ == "__main__":
    # Similarity example
    text = "After stealing money from the bank vault, " \
           "the bank robber was seen fishing on the Mississippi river bank."
    bert = PretrainedBert('../../../clinical_kb_albert')
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
        diff_bank = 1 - cosine(embeds[10], embeds[19])
        same_bank = 1 - cosine(embeds[10], embeds[6])
        print('First 5 vector values for each instance of "bank".')
        print("[bank] vault:\t", str(embeds[6][:5]))   # "bank" at position 6 is contextually closer to at 10
        print("[bank] robber:\t", str(embeds[10][:5]))  # and contextually far apart from at 19
        print("river [bank]:\t", str(embeds[19][:5]))
        print('Vector similarity for  *similar*  meanings:  %.2f' % same_bank)
        print('Vector similarity for *different* meanings:  %.2f' % diff_bank)
        print("------------------")
