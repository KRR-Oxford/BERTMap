"""
OntoInvertedIndex class with as entries the subword tokens retrieved from ontological class texts.
"""

from transformers import AutoTokenizer
from bertmap.onto import OntoText
from collections import defaultdict
from itertools import chain
import json, os, re
from typing import List, Optional


class OntoInvertedIndex:
    
    def __init__(self, 
                 ontotext: Optional[OntoText]=None, 
                 tokenizer_path: Optional[str]=None, 
                 cut: int=0, 
                 index_file: Optional[str]=None):
        
        self.cut = cut
        if index_file: 
            self.load_index(index_file)
        else:
            self.ontotext = ontotext
            self.tokenizer_path = tokenizer_path
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.construct_index(cut, *self.ontotext.properties)

    def __repr__(self):
        return f"<OntoInvertedIndex num_entries={len(self.index)} cut={self.cut} tokenizer_path={self.tokenizer_path}>"
        
    def tokenize(self, texts) -> List[str]:
        return chain.from_iterable([self.tokenizer.tokenize(text) for text in texts])
            
    def construct_index(self, cut: int, *properties: str) -> None:
        """Create Inverted Index with sub-word tokens

        Args:
            cut (int): ignore sub-word tokens of length <= cut
        """
        self.index = defaultdict(list)
        for cls_iri, text_dict in self.ontotext.texts.items():
            for prop, texts in text_dict.items():
                if not prop in properties: continue
                tokens = self.tokenize(texts)
                for tk in tokens:
                    if len(tk) > cut: self.index[tk].append(self.ontotext.class2idx[cls_iri])

    def save_index(self, index_file: str) -> None:
        with open(index_file, "w") as f:
            json.dump(self.index, f, indent=4, separators=(',', ': '), sort_keys=True)
    
    def load_index(self, index_file: str) -> None:
        with open(index_file, "r") as f:
            self.index = json.load(f)
        file = index_file.split("/")[-1]
        if os.path.exists(index_file.replace(file, "info")):
            with open(index_file.replace(file, "info"), "r") as f:
                self.tokenizer_path = re.findall(r"tokenizer_path=(.+)>", f.readlines()[2])[0]
            print(f"load tokenizer from {self.tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        else: self.tokenizer_path = "missing"
