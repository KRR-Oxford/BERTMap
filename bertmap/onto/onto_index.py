"""
Inverted Index built on WordPiece tokenizer (subword level)
"""

from transformers import AutoTokenizer
from bertmap.onto import Ontology
from collections import defaultdict
from itertools import chain


class OntoInvertedIndex:
    
    def __init__(self, tokenizer_path="emilyalsentzer/Bio_ClinicalBERT"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.index = defaultdict(list)
        
    def tokenize_class_text(self, class_text: str):
        texts, _ = Ontology.parse_class_text(class_text)
        return chain.from_iterable([self.tokenizer.tokenize(text) for text in texts])
            
    def construct_index(self, onto_name, onto_class2text_file, cut=0, clear=False):
        self.onto_name = onto_name
        if clear:
            self.index = defaultdict(list)
        self.class2text = Ontology.load_class2text(onto_class2text_file) \
            if type(onto_class2text_file) is str else onto_class2text_file
        for i, dp in self.class2text.iterrows():
            tokens = self.tokenize_class_text(dp["Class-Text"])
            for tk in tokens:
                if len(tk) > cut:
                    self.index[tk].append(i)
        print(f"Sub-word level Inverted Index built for ontology {self.onto_name}, with token length cut at {cut}.")

            
    