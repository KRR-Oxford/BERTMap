from bertmap.onto import Ontology
from bertmap.corpora import OntologyCorpus
from bertmap.utils.oaei_utils import read_tsv_mappings
from collections import OrderedDict
from itertools import product

class CrossOntoCorpus(OntologyCorpus):
    
    def __init__(self, src_onto_path, tgt_onto_path, known_mappings_tsv, 
                 src_onto_class2text_tsv=None, tgt_onto_class2text_tsv=None, 
                 properties=["label"], corpus_path=None):
        self.src_ontology = Ontology(src_onto_path)
        self.tgt_ontology = Ontology(tgt_onto_path)
        self.src_onto_class2text = Ontology.load_class2text(src_onto_class2text_tsv) if src_onto_class2text_tsv \
            else self.src_ontology.create_class2text(*properties) 
        self.tgt_onto_class2text = Ontology.load_class2text(tgt_onto_class2text_tsv) if tgt_onto_class2text_tsv \
            else self.tgt_ontology.create_class2text(*properties)
        self.known_mappings = read_tsv_mappings(known_mappings_tsv)
        self.corpus_names = ["forward_synonyms", "backward_synonyms"]
        self.onto_name = self.src_ontology.iri_abbr.replace(":", "") + "2" + self.tgt_ontology.iri_abbr.replace(":", "")
        super().__init__(corpus_path=corpus_path)
        
            
    def create_corpora(self):
        self.src_onto_class2text_dict = self.create_class2text_dict(self.src_onto_class2text)
        self.tgt_onto_class2text_dict = self.create_class2text_dict(self.tgt_onto_class2text)
        self.cross_onto_synonyms()
    
    def train_val_split(self, corpus_names):
        return super().train_val_split(corpus_names)
        
    @staticmethod
    def create_class2text_dict(class2text):
        class2text_dict = OrderedDict()
        for _, dp in class2text.iterrows():
            class2text_dict[dp["Class-IRI"]] = dp["Class-Text"]
        return class2text_dict
        
    def cross_onto_synonyms(self):
        forward = []
        backward = []
        for ref_map in self.known_mappings:
            src_class, tgt_class = ref_map.split("\t")
            src_labels, _ = Ontology.parse_class_text(self.src_onto_class2text_dict[src_class])
            tgt_labels, _ = Ontology.parse_class_text(self.tgt_onto_class2text_dict[tgt_class])
            for_list = list(product(src_labels, tgt_labels))
            back_list = [(y, x, 1) for (x, y) in for_list]
            for_list = [(x, y, 1) for (y, x, _) in back_list]
            forward += for_list
            backward += back_list
        self.forward_synonyms = forward
        self.backward_synonyms = backward
