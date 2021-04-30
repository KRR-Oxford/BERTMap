"""
OntoInvertedIndex class with as entries the subword tokens retrieved from ontological class texts.
"""

from transformers import AutoTokenizer
from bertmap.onto import OntoBox
from collections import defaultdict
from itertools import chain
import json


class OntoInvertedIndex:
    
    def __init__(self, ontobox: OntoBox, tokenizer_path: str, 
                 cut=0, properties=["label"], index_file=None):
        self.ontobox = ontobox
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.cut = cut
        if index_file: self.load_index(index_file)
        else: self.construct_index(cut, *properties)

    def __repr__(self):
        report = "<OntoInvertedIndex>:\n"
        report += f"\t<entry num={len(self.index)} cut={self.cut}>\n"
        report += f"\n{str(self.ontobox)}".replace("\n", "\n\t")
        report += f"\n</OntoInvertedIndex>\n"
        return report
        
    def tokenize(self, texts):
        return chain.from_iterable([self.tokenizer.tokenize(text) for text in texts])
            
    def construct_index(self, cut: int, *properties):
        """Create Inverted Index with sub-word tokens

        Args:
            cut (int): ignore sub-word tokens of length <= cut
        """
        self.index = defaultdict(list)
        # default lexicon information is the "labels"
        if not properties: properties = ["label"]
        for cls_iri, text_dict in self.ontobox.classtexts.items():
            for prop, texts in text_dict.items():
                if not prop in properties: continue
                tokens = self.tokenize(texts)
                for tk in tokens:
                    if len(tk) > cut: self.index[tk].append(self.ontobox.class2idx[cls_iri])

    def save_index(self, index_file):
        with open(index_file, "w") as f:
            json.dump(self.index, f, indent=4, separators=(',', ': '), sort_keys=True)
    
    def load_index(self, index_file):
        with open(index_file, "r") as f:
            self.index = json.load(f)
