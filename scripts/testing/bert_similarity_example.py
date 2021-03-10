import sys
sys.path.append("/home/yuahe/projects/OntoAlign-py")
from ontoalign.embeds import PretrainedBert
from ontoalign.onto import OntoMetric

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
    diff_bank = 1 - OntoMetric.cos_dist(embeds[10], embeds[19])  # it's equal to cos(a, b)
    same_bank = 1 - OntoMetric.cos_dist(embeds[10], embeds[6])
    print('First 5 vector values for each instance of "bank".')
    print("[bank] vault:\t", str(embeds[6][:5]))   # "bank" at position 6 is contextually closer to at 10
    print("[bank] robber:\t", str(embeds[10][:5]))  # and contextually far apart from at 19
    print("river [bank]:\t", str(embeds[19][:5]))
    print('Vector similarity for  *similar*  meanings:  %.2f' % same_bank)
    print('Vector similarity for *different* meanings:  %.2f' % diff_bank)
    print("------------------")