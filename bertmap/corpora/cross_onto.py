"""
Ontology Corpus class from known/seed cross-ontology mappings.
"""

from bertmap.onto import Ontology
from bertmap.corpora import OntologyCorpus
from bertmap.utils.oaei_utils import read_tsv_mappings
from bertmap.utils import uniqify, exclude_randrange
from collections import OrderedDict, defaultdict
from copy import deepcopy
import random

class CrossOntoCorpus(OntologyCorpus):
    
    def __init__(self, onto_name, src_onto_path, tgt_onto_path, known_mappings_tsv, 
                 src_onto_class2text_tsv=None, tgt_onto_class2text_tsv=None, 
                 properties=["label"], sample_rate=5, corpus_path=None):
        self.corpus_type = "cross-onto"
        super().__init__(src_onto_path, tgt_onto_path, known_mappings_tsv, src_onto_class2text_tsv, tgt_onto_class2text_tsv, 
                         properties, sample_rate, onto_name=onto_name, corpus_path=corpus_path)
        
    def init_config(self, src_onto_path, tgt_onto_path, known_mappings_tsv, 
                    src_onto_class2text_tsv=None, tgt_onto_class2text_tsv=None, 
                    properties=["label"], sample_rate=5):
        self.src_ontology = Ontology(src_onto_path)
        self.tgt_ontology = Ontology(tgt_onto_path)
        self.src_onto_class2text = Ontology.load_class2text(src_onto_class2text_tsv) if src_onto_class2text_tsv \
            else self.src_ontology.create_class2text(*properties) 
        self.tgt_onto_class2text = Ontology.load_class2text(tgt_onto_class2text_tsv) if tgt_onto_class2text_tsv \
            else self.tgt_ontology.create_class2text(*properties)
        self.known_mappings = read_tsv_mappings(known_mappings_tsv)
        
        self.corpus_dict = defaultdict(lambda:self.term_dict())
        self.sample_rate = sample_rate
            
    def create_corpus(self):
        self.src_onto_class2text_dict = self.create_class2text_dict(self.src_onto_class2text)
        self.tgt_onto_class2text_dict = self.create_class2text_dict(self.tgt_onto_class2text)
        print(f"Generating the Cross-Ontology Corpora for: {self.onto_name}.")
        self.cross_onto_synonyms()
        self.cross_onto_nonsynonyms(sample_rate=self.sample_rate)
        self.corpus_dict[" corpus_info "] = {"corpus_type": "Cross-ontology Corpus", 
                                             "corpus_onto": self.onto_name, 
                                             "synonyms": self.synonym_count,
                                             "soft_nonsynonyms": self.nonsynonym_count,
                                             "hard_nonsynonyms": "Not Available",
                                             "num_violated": len(self.violation)}
        self.report(self.corpus_dict)
        
    @staticmethod
    def create_class2text_dict(class2text):
        class2text_dict = OrderedDict()
        for _, dp in class2text.iterrows():
            class2text_dict[dp["Class-IRI"]] = dp["Class-Text"]
        return class2text_dict
        
    def cross_onto_synonyms(self):
        self.synonym_count = 0
        for i in range(len(self.known_mappings)):
            ref_map = self.known_mappings[i]
            src_class, tgt_class = ref_map.split("\t")
            src_labels, _ = Ontology.parse_class_text(self.src_onto_class2text_dict[src_class])
            tgt_labels, _ = Ontology.parse_class_text(self.tgt_onto_class2text_dict[tgt_class])
            for src_label in src_labels:
                src_term_dict = self.corpus_dict[src_label]
                src_term_synonyms = deepcopy(src_term_dict["synonyms"])
                existed_num = len(src_term_synonyms)
                src_term_synonyms += tgt_labels
                src_term_synonyms = uniqify(src_term_synonyms)
                self.synonym_count += len(src_term_synonyms) - existed_num
                ##### Update the dictionary #####
                src_term_dict["synonyms"] = src_term_synonyms
                self.corpus_dict[src_label] = src_term_dict
 
    def cross_onto_nonsynonyms(self, sample_rate=5):
        """There is no hard nonsynonym available on the cross-ontology level.
           Here the nonsynonyms are extracted from randomly mis-aligned classes.
        """
        self.nonsynonym_count = 0
        self.violation = []
        for i in range(len(self.known_mappings)):
            ref_map = self.known_mappings[i]
            src_class, tgt_class = ref_map.split("\t")
            src_labels, _ = Ontology.parse_class_text(self.src_onto_class2text_dict[src_class])
            tgt_labels, _ = Ontology.parse_class_text(self.tgt_onto_class2text_dict[tgt_class])
            self.negative_sampling_from_maps(src_labels, i, sample_rate=sample_rate, search_from="TGT")
            self.negative_sampling_from_maps(tgt_labels, i, sample_rate=sample_rate, search_from="SRC")

    def negative_sampling_from_maps(self, class_labels, map_ind, sample_rate=5, search_from="SRC"):
        """The helper function for sampling R negative labels from opposite ontology O' for each class label of the ontology O"""
        if search_from == "SRC":
            class2text_dict = self.src_onto_class2text_dict 
        elif search_from == "TGT":
            class2text_dict = self.tgt_onto_class2text_dict
        else:
            raise TypeError
        
        for label in class_labels:
            neg_map_inds = [exclude_randrange(0, len(self.known_mappings), exclude=map_ind) for _ in range(sample_rate-1)]
            label_dict = self.corpus_dict[label]
            label_nonsynonyms = label_dict["soft_nonsynonyms"]
            existed_num = len(label_nonsynonyms)
            for nid in neg_map_inds:
                neg_map = self.known_mappings[nid]
                src_neg_class, tgt_neg_class = neg_map.split("\t")
                neg_class = src_neg_class if search_from == "SRC" else tgt_neg_class
                # assert neg_class in class2text_dict.keys()  # simple test to ensure that the extracted class coming from the opposite ontology
                neg_label_list, neg_label_num = Ontology.parse_class_text(class2text_dict[neg_class])
                neg_label_ind = random.randrange(0, neg_label_num)
                neg_label = neg_label_list[neg_label_ind]
                if self.negative_sample_check(label, neg_label):
                    label_nonsynonyms.append(neg_label)
                else:
                    self.violation.append((label, neg_label))
            label_nonsynonyms = uniqify(label_nonsynonyms)
            self.nonsynonym_count += len(label_nonsynonyms) - existed_num
            ###### update the dictionary ######
            label_dict["soft_nonsynonyms"] = label_nonsynonyms
            self.corpus_dict[label] = label_dict       
