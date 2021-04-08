"""
Ontology Corpus class from known/seed cross-ontology mappings.
"""

from bertmap.onto import Ontology
from bertmap.corpora import OntologyCorpus
from bertmap.utils.oaei_utils import read_tsv_mappings
from bertmap.utils import uniqify, exclude_randrange
from collections import OrderedDict
from itertools import product
import random
import pandas as pd

class CrossOntoCorpus(OntologyCorpus):
    
    def __init__(self, src_onto_path, tgt_onto_path, known_mappings_tsv, 
                 src_onto_class2text_tsv=None, tgt_onto_class2text_tsv=None, 
                 properties=["label"], sample_rate=5, corpus_path=None):
        self.src_ontology = Ontology(src_onto_path)
        self.tgt_ontology = Ontology(tgt_onto_path)
        self.src_onto_class2text = Ontology.load_class2text(src_onto_class2text_tsv) if src_onto_class2text_tsv \
            else self.src_ontology.create_class2text(*properties) 
        self.tgt_onto_class2text = Ontology.load_class2text(tgt_onto_class2text_tsv) if tgt_onto_class2text_tsv \
            else self.tgt_ontology.create_class2text(*properties)
        self.known_mappings = read_tsv_mappings(known_mappings_tsv)
        self.corpus_names = ["forward_synonyms", "backward_synonyms"]
        self.sample_rate = sample_rate
        self.onto_name = self.src_ontology.iri_abbr.replace(":", "") + "2" + self.tgt_ontology.iri_abbr.replace(":", "")
        super().__init__(corpus_path=corpus_path)
        
            
    def create_corpora(self):
        self.src_onto_class2text_dict = self.create_class2text_dict(self.src_onto_class2text)
        self.tgt_onto_class2text_dict = self.create_class2text_dict(self.tgt_onto_class2text)
        print(f"Generating the Cross-Ontology Corpora for: {self.onto_name}.")
        self.cross_onto_synonyms()
        # self.cross_onto_nonsynonyms(sample_rate=self.sample_rate)
        
    @staticmethod
    def create_class2text_dict(class2text):
        class2text_dict = OrderedDict()
        for _, dp in class2text.iterrows():
            class2text_dict[dp["Class-IRI"]] = dp["Class-Text"]
        return class2text_dict
        
    def cross_onto_synonyms(self):
        synonyms = []
        for i in range(len(self.known_mappings)):
            ref_map = self.known_mappings[i]
            src_class, tgt_class = ref_map.split("\t")
            src_labels, _ = Ontology.parse_class_text(self.src_onto_class2text_dict[src_class])
            tgt_labels, _ = Ontology.parse_class_text(self.tgt_onto_class2text_dict[tgt_class])
            for_list = list(product(src_labels, tgt_labels))
            formatted_list = [(" | ".join([x, y]), " | ".join([y, x]), 1, i) for x, y in for_list]
            synonyms += formatted_list
        print(f"---------- Raw # Synonym Pairs ----------")
        print(f"[for/back]: {len(synonyms)}")
        synonyms = uniqify(synonyms)
        print("---------- No Dups # Synonym Pairs ----------")
        print(f"[for/back]: {len(synonyms)}")
        self.synonyms = pd.DataFrame(synonyms, columns=["Forward", "Backward", "Synonymous", "Map-Ind"])
        print("--------------- Example Pairs --------------")
        for _ in range(2):
            exp_ind = random.randrange(0, len(synonyms))
            print(f"[{exp_ind}]\n\t[for/back]: {synonyms[exp_ind]}")
        print("\n")
        
    def cross_onto_nonsynonyms(self, sample_rate=5):
        """There is no hard nonsynonym available on the cross-ontology level.
           Here the nonsynonyms are extracted from randomly mis-aligned classes.
        """
        nonsynonyms = []
        for i in range(len(self.known_mappings)):
            ref_map = self.known_mappings[i]
            src_class, tgt_class = ref_map.split("\t")
            src_labels, _ = Ontology.parse_class_text(self.src_onto_class2text_dict[src_class])
            tgt_labels, _ = Ontology.parse_class_text(self.tgt_onto_class2text_dict[tgt_class])
            nonsynonyms += self.negative_sampling(src_labels, i, sample_rate=sample_rate, search_from="TGT")
            nonsynonyms += self.negative_sampling(tgt_labels, i, sample_rate=sample_rate, search_from="SRC")
        print("---------- Raw # Nonsynonym Pairs ----------")
        print(f"[for/back]: {len(nonsynonyms)}")
        nonsynonyms = uniqify(nonsynonyms)
        print("---------- No Dups # Nonsynonym Pairs ----------")
        print(f"[for/back]: {len(nonsynonyms)}")
        self.nonsynonyms = pd.DataFrame(nonsynonyms, columns=["Forward", "Backward", "Synonymous", "Map-Ind"])
        print("--------------- Example Pairs --------------")
        for _ in range(2):
            exp_ind = random.randrange(0, len(nonsynonyms))
            print(f"[{exp_ind}]\n\t[for/back]: {nonsynonyms[exp_ind]}")
        print("\n")

    def negative_sampling(self, class_labels, map_ind, sample_rate=5, search_from="SRC"):
        """The helper function for sampling R negative labels from opposite ontology O' for each class label of the ontology O"""
        if search_from == "SRC":
            class2text_dict = self.src_onto_class2text_dict 
        elif search_from == "TGT":
            class2text_dict = self.tgt_onto_class2text_dict
        else:
            raise TypeError
        
        partial_nonsynonyms = []
        for label in class_labels:
            neg_map_inds = [exclude_randrange(0, len(self.known_mappings), exclude=map_ind) for _ in range(sample_rate-1)]
            for nid in neg_map_inds:
                neg_map = self.known_mappings[nid]
                src_neg_class, tgt_neg_class = neg_map.split("\t")
                neg_class = src_neg_class if search_from == "SRC" else tgt_neg_class
                # assert neg_class in class2text_dict.keys()  # simple test to ensure that the extracted class coming from the opposite ontology
                neg_label_list, neg_label_num = Ontology.parse_class_text(class2text_dict[neg_class])
                neg_label_ind = random.randrange(0, neg_label_num)
                neg_label = neg_label_list[neg_label_ind]
                src_label = neg_label if search_from == "SRC" else label
                tgt_label = neg_label if search_from == "TGT" else label
                if src_label == tgt_label:
                    print(label, neg_label, map_ind, nid)
                    print(self.known_mappings[map_ind])
                    print(self.known_mappings[nid])
                    print(class2text_dict[neg_class])
                assert src_label != tgt_label
                partial_nonsynonyms.append((" | ".join([src_label, tgt_label]), " | ".join([tgt_label, src_label]), 0, f"{map_ind}-{nid}"))
        
        return partial_nonsynonyms         
