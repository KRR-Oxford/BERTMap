from bertmap.onto import Ontology
from bertmap.utils import uniqify, ex_randrange
import random
import itertools
import pandas as pd


class IntraOntoCorpus:
    
    def __init__(self, onto_path, onto_lexicon_tsv=None, properties=["label"], corpus_path=None):
        
        self.ontology = Ontology(onto_path)
        self.lexicon = Ontology.load_iri_lexicon_file(onto_lexicon_tsv) if onto_lexicon_tsv \
            else self.ontology.iri_lexicon_df(self, *properties) 
        self.corpus_names = ["id_synonyms", "forward_synonyms", "backward_synonyms", 
                             "forward_soft_nonsymnonyms", "backward_soft_nonsynonyms",
                             "forward_hard_nonsymnonyms", "backward_hard_nonsynonyms"]
            
        # form the intra-ontology corpus
        if not corpus_path:
            self.intra_onto_synonyms()
            self.intra_onto_soft_nonsynonyms()
            self.intra_onto_hard_nonsynonyms()
        else:
            # load corpus from local storage
            self.load_corpora(save_dir=corpus_path)
        
        
    def intra_onto_synonyms(self):
        identity, forward, backward = [], [], []  # (a_i, a_i); (a_i, a_{j>i}); (a_i, a_{j<i})
        for _, dp in self.lexicon.iterrows():
            lexicon = dp["Entity-Lexicon"]
            alias_list, num = Ontology.parse_entity_lexicon(lexicon)
            identity += list(zip(alias_list, alias_list))
            for i in range(num):
                for j in range(i+1, num):
                    forward.append((alias_list[i], alias_list[j]))
                    backward.append((alias_list[j], alias_list[i]))    
                     
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
        for i, dp in self.lexicon.iterrows():
            lexicon = dp["Entity-Lexicon"]
            label_list, _ = Ontology.parse_entity_lexicon(lexicon)
            for label in label_list:
                # for each label, sample X (sample rate) random negatives
                neg_class_inds = [ex_randrange(0, len(self.lexicon), ex=i) for _ in range(sample_rate-1)]
                for nid in neg_class_inds:
                    neg_label_list, neg_label_num = Ontology.parse_entity_lexicon(self.lexicon.iloc[nid]["Entity-Lexicon"])
                    neg_label_ind = random.randrange(0, neg_label_num)
                    neg_label = neg_label_list[neg_label_ind]
                    forward.append((label, neg_label))
                    backward.append((neg_label, label))
        print("---------- Raw # Soft Nonsynonym Pairs ----------")
        print(f"[for]: {len(forward)}; [back]: {len(backward)}")
        forward, backward = uniqify(forward), uniqify(backward)
        print("---------- No Dups # Soft Nonsynonym Pairs ----------")
        print(f"[for]: {len(forward)}; [back]: {len(backward)}")
        self.forward_soft_nonsymnonyms = forward
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
                encoded_labels = Ontology.encode_entity_lexicon(scl, "label")
                label_list, _ = Ontology.parse_entity_lexicon(encoded_labels)
                sib_labels.append(label_list)
            # e.g. sibling1: ["a", "b"]; sibling2: ["c"] -> [("a", "c"), ("b", "c")] as forward 
            for i in range(len(sib_labels)):
                for j in range(i+1, len(sib_labels)):
                    for_list = list(itertools.product(sib_labels[i], sib_labels[j]))
                    back_list = [(y, x) for (x, y) in for_list]
                    forward += for_list
                    backward += back_list
        print("---------- Raw # Hard Nonsynonym Pairs ----------")
        print(f"[for]: {len(forward)}; [back]: {len(backward)}")
        forward, backward = uniqify(forward), uniqify(backward)
        print("---------- No Dups # Hard Nonsynonym Pairs ----------")
        print(f"[for]: {len(forward)}; [back]: {len(backward)}")
        self.forward_hard_nonsymnonyms = forward
        self.backward_hard_nonsynonyms = backward
        print("--------------- Example Pairs --------------")
        for _ in range(2):
            exp_ind = random.randrange(0, len(forward))
            print(f"[{exp_ind}]\n\t[for]: {forward[exp_ind]}\n\t[back]: {backward[exp_ind]}")
        print("\n")
        
    def save_corpora(self, save_dir):
        for name in self.corpus_names:
            corpus = getattr(self, name)
            df = pd.DataFrame(corpus, columns=["Label1", "Label2"])
            onto_name = self.ontology.iri_abbr.replace(":", "")
            df.to_csv(f"{save_dir}/{onto_name}.{name}.tsv", sep='\t', index=False)
            
    def load_corpora(self, save_dir):
        print("--------------- Loaded Corpora Sizes --------------")
        for name in self.corpus_names:
            onto_name = self.ontology.iri_abbr.replace(":", "")
            setattr(self, name, pd.read_csv(f"{save_dir}/{onto_name}.{name}.tsv", sep='\t'))
            tag = " ".join(name.split("_"))
            print(f"{tag}: {len(getattr(self, name))}")
        print("---------------------------------------------------")