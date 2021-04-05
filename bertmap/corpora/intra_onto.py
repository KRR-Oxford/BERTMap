from bertmap.onto import Ontology
from bertmap.utils import uniqify, ex_randrange
from bertmap.corpora import OntologyCorpus
import random
import itertools
import pandas as pd


class IntraOntoCorpus(OntologyCorpus):
    
    def __init__(self, onto_path, onto_class2text_tsv=None, properties=["label"], corpus_path=None):
        self.ontology = Ontology(onto_path)
        self.class2text = Ontology.load_class2text(onto_class2text_tsv) if onto_class2text_tsv \
            else self.ontology.create_class2text(*properties) 
        self.corpus_names = ["id_synonyms", "forward_synonyms", "backward_synonyms", 
                             "forward_soft_nonsynonyms", "backward_soft_nonsynonyms",
                             "forward_hard_nonsynonyms", "backward_hard_nonsynonyms"]
        self.onto_name = self.ontology.iri_abbr.replace(":", "")
        super().__init__(corpus_path=corpus_path)
            
    def create_corpora(self):
        self.intra_onto_synonyms()
        self.intra_onto_soft_nonsynonyms()
        self.intra_onto_hard_nonsynonyms()
    
    def train_val_split(self, corpus_names):
        pos_corpora = []
        soft_neg_corpora = []
        hard_neg_corpora = []
        for cn in corpus_names:
            if "soft" in cn and "nonsynonyms" in cn:
                soft_neg_corpora.append(getattr(self, cn))
            elif "hard" in cn and "nonsynonyms" in cn:
                hard_neg_corpora.append(getattr(self, cn))
            else:
                pos_corpora.append(getattr(self, cn))
        # for each positive sample, retrieve 1 soft and 1 hard negatives
        pos_data = pd.concat(pos_corpora, ignore_index=True)
        soft_neg_data = pd.concat(soft_neg_corpora, ignore_index=True).sample(len(pos_data))
        hard_neg_data = pd.concat(hard_neg_corpora, ignore_index=True).sample(len(pos_data))
        data = pd.concat([pos_data, soft_neg_data, hard_neg_data], ignore_index=True).sample(frac=1).reset_index(drop=True)
        val_data = data.sample(frac=0.2)
        train_data = data.drop(val_data.index)
        return train_data, val_data
        
    def intra_onto_synonyms(self):
        identity, forward, backward = [], [], []  # (a_i, a_i); (a_i, a_{j>i}); (a_i, a_{j<i})
        for _, dp in self.class2text.iterrows():
            lexicon = dp["Entity-Lexicon"]
            alias_list, num = Ontology.parse_class_text(lexicon)
            identity += list(zip(alias_list, alias_list, [1]*len(alias_list)))
            for i in range(num):
                for j in range(i+1, num):
                    forward.append((alias_list[i], alias_list[j], 1))
                    backward.append((alias_list[j], alias_list[i], 1))    
                     
        print("---------- Raw # Synonym Pairs ----------")
        print(f"[id]: {len(identity)}; [for]: {len(forward)}; [back]: {len(backward)}")
        identity, forward, backward = uniqify(identity), uniqify(forward), uniqify(backward)
        print("---------- No Dups # Synonym Pairs ----------")
        print(f"[id]: {len(identity)}; [for]: {len(forward)}; [back]: {len(backward)}")
        self.id_synonyms, self.forward_synonyms, self.backward_synonyms = identity, forward, backward
        print("--------------- Example Pairs --------------")
        exp_ind = random.randrange(0, len(identity))
        print(f"[{exp_ind}]\n\t[id]: {identity[exp_ind]}")
        for _ in range(2):
            exp_ind = random.randrange(0, len(identity))
            print(f"[{exp_ind}]\n\t[for]: {forward[exp_ind]}\n\t[back]: {backward[exp_ind]}")
        print("\n")
    
    def intra_onto_soft_nonsynonyms(self, sample_rate=10): 
        forward, backward = [], []
        for i, dp in self.class2text.iterrows():
            lexicon = dp["Entity-Lexicon"]
            label_list, _ = Ontology.parse_class_text(lexicon)
            for label in label_list:
                # for each label, sample X (sample rate) random negatives
                neg_class_inds = [ex_randrange(0, len(self.class2text), ex=i) for _ in range(sample_rate-1)]
                for nid in neg_class_inds:
                    neg_label_list, neg_label_num = Ontology.parse_class_text(self.class2text.iloc[nid]["Entity-Lexicon"])
                    neg_label_ind = random.randrange(0, neg_label_num)
                    neg_label = neg_label_list[neg_label_ind]
                    forward.append((label, neg_label, 0))
                    backward.append((neg_label, label, 0))
        print("---------- Raw # Soft Nonsynonym Pairs ----------")
        print(f"[for]: {len(forward)}; [back]: {len(backward)}")
        forward, backward = uniqify(forward), uniqify(backward)
        print("---------- No Dups # Soft Nonsynonym Pairs ----------")
        print(f"[for]: {len(forward)}; [back]: {len(backward)}")
        self.forward_soft_nonsynonyms = forward
        self.backward_soft_nonsynonyms = backward
        print("--------------- Example Pairs --------------")
        for _ in range(2):
            exp_ind = random.randrange(0, len(forward))
            print(f"[{exp_ind}]\n\t[for]: {forward[exp_ind]}\n\t[back]: {backward[exp_ind]}")  
        print("\n")
            
            
    def intra_onto_hard_nonsynonyms(self):
        classes = list(self.ontology.onto.classes())   
        forward, backward = [], []
        for cl in classes:
            subcls = list(cl.subclasses())
            if len(subcls) <= 1:
                continue
            # with at least two sibiling classes we can extract hard negatives
            sib_labels = []
            for scl in subcls:
                encoded_labels = Ontology.encode_class_text(scl, "label")
                label_list, _ = Ontology.parse_class_text(encoded_labels)
                sib_labels.append(label_list)
            # e.g. sibling1: ["a", "b"]; sibling2: ["c"] -> [("a", "c"), ("b", "c")] as forward 
            for i in range(len(sib_labels)):
                for j in range(i+1, len(sib_labels)):
                    for_list = list(itertools.product(sib_labels[i], sib_labels[j]))
                    back_list = [(y, x, 0) for (x, y) in for_list]
                    for_list = [(x, y, 0) for (y, x, _) in back_list]
                    forward += for_list
                    backward += back_list
        print("---------- Raw # Hard Nonsynonym Pairs ----------")
        print(f"[for]: {len(forward)}; [back]: {len(backward)}")
        forward, backward = uniqify(forward), uniqify(backward)
        print("---------- No Dups # Hard Nonsynonym Pairs ----------")
        print(f"[for]: {len(forward)}; [back]: {len(backward)}")
        self.forward_hard_nonsynonyms = forward
        self.backward_hard_nonsynonyms = backward
        print("--------------- Example Pairs --------------")
        for _ in range(2):
            exp_ind = random.randrange(0, len(forward))
            print(f"[{exp_ind}]\n\t[for]: {forward[exp_ind]}\n\t[back]: {backward[exp_ind]}")
        print("\n")
