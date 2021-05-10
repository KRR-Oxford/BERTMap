"""
OntoInvertedIndex class with as entries the subword tokens retrieved from ontological class texts.
"""

import json
from collections import defaultdict
from itertools import chain
from typing import List, Optional

from bertmap.onto import OntoText
from transformers import AutoTokenizer


class OntoInvertedIndex:
    def __init__(
        self,
        ontotext: Optional[OntoText] = None,
        tokenizer_path: Optional[str] = None,
        cut: int = 0,
        index_file: Optional[str] = None,
    ):

        self.cut = cut
        if index_file:
            self.load_index(index_file)
        else:
            self.ontotext = ontotext
            self.set_tokenizer(tokenizer_path)
            self.construct_index(cut, *self.ontotext.properties)

    def __repr__(self):
        return f"<OntoInvertedIndex num_entries={len(self.index)} cut={self.cut} tokenizer_path={self.tokenizer_path}>"

    def set_tokenizer(self, tokenizer_path: str) -> None:
        """Set or change the tokenizer used for creating the Index,
        note that when loading, the tokenizer is not loaded by default,
        but in the OntoBox class, the tokenizer info is stored
        """
        self.tokenizer_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

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
                if not prop in properties:
                    continue
                tokens = self.tokenize(texts)
                for tk in tokens:
                    if len(tk) > cut:
                        self.index[tk].append(self.ontotext.class2idx[cls_iri])

    def save_index(self, index_file: str) -> None:
        with open(index_file, "w") as f:
            json.dump(self.index, f, indent=4, separators=(",", ": "), sort_keys=True)

    def load_index(self, index_file: str) -> None:
        with open(index_file, "r") as f:
            self.index = json.load(f)
