"""
Ontology Corpus superclass, it requires implementation of how to create corpora from the classtexts.

The corpus essentially stores pairs of texts that are semantically related, including synonyms and nonsynonyms:

    Synonyms are divided into *forward*, *backward* and *identity* synonyms;
    Nonsynonyms are divided into *soft* (forward/backward) and *hard* (forward/backward) nonsynonyms.
    
    {Identity synonyms} are pairs of duplicated classtexts -> to learn Reflexivity or String matching.

    {Forward synonyms} are pairs of *distinct* classtexts belong to one class, while {backward synonyms} are just the reversed
        -> to learn Symmertry and Transitiviy.
        
    {Soft nonsynonyms} are pairs of classtexts from random choices of different classes. 
    
    {Hard nonsynonyms} are pairs of classtexts from *disjoint* classes (e.g. sibling classes up to certain depth)
    
    The concepts of forward and backward apply to nonsynonyms as well.

    Data structure illustration:
    {   
        ...,
        class_label:{ 
            synonyms: []
            soft_nonsynonyms: []
            hard_nonsynonyms: []
        },
        ...
    }
"""

import json
from typing import Optional

import pandas as pd


class OntoCorpus:
    
    def __init__(self, *corpus_args, corpus_file: Optional[str]=None):
        if corpus_file: self.load_corpus(corpus_file)
        else: self.from_saved = False; self.config(*corpus_args); self.create_corpus()
        assert self.corpus_type is not None
        assert self.corpus_info is not None
        assert self.corpus is not None
        
    def config(self, *config_args):
        raise NotImplementedError
        
    def create_corpus(self):
        raise NotImplementedError
    
    def __repr__(self):
        corpus_info = self.corpus[" corpus_info "]
        onto = corpus_info["onto"]
        report = f"<OntologyCorpus type={self.corpus_type} onto={onto}>\n"
        synonym_report = "\t<Synonyms "
        for k, v in corpus_info["synonyms"].items():
            synonym_report += f"{k}={v} "
        synonym_report = synonym_report.rstrip()
        synonym_report += ">\n"
        nonsynonym_report = "\t<Nonsynonyms "
        for k, v in corpus_info["nonsynonyms"].items():
            nonsynonym_report += f"{k}={v} "
        nonsynonym_report = nonsynonym_report.rstrip()
        nonsynonym_report += ">\n"
        report += synonym_report
        report += nonsynonym_report
        if corpus_info["others"]:
            others_report = "\t<OtherInfo "
            for k, v in corpus_info["others"].items():
                others_report += f"{k}={v} "
            others_report = others_report.rstrip()
            others_report += ">\n"
            report += others_report
        report += "</OntologyCorpus>"
        return report

    @staticmethod
    def init_semantic_dict():
        # some corpus might not define the hard nonsynonyms so that they will be empty
        return {"synonyms": [], "soft_nonsynonyms": [], "hard_nonsynonyms": []}
    
    def init_corpus_info(self):
        """init corpus information storage
        """
        return {
            "type": self.corpus_type,
            "onto": [],
            "synonyms": {"non-id": 0, "id": None},
            "nonsynonyms": {"soft": 0, "hard": None, "soft-back": None, "hard-back": None, "removed_violations": 0},
            "others": {}
        }

    def get_identity_synonyms(self):
        """[Reflexivity]: Map each classtext to itself
        """
        classtexts = list(self.corpus.keys())[1:]  
        return list(zip(classtexts, classtexts, [1]*len(classtexts)))
    
    def extract_semantic_pairs(self):
        """Extract classtext pairs from corpus
        """
        synonyms = []
        soft_nonsynonyms = []
        hard_nonsynonyms = []
        for label, semantic_dict in self.corpus.items():
            if label == " corpus_info ": continue
            synonyms += [(label, s, 1) for s in semantic_dict["synonyms"]]
            soft_nonsynonyms += [(label, ns, 0) for ns in semantic_dict["soft_nonsynonyms"]]
            hard_nonsynonyms += [(label, ns, 0) for ns in semantic_dict["hard_nonsynonyms"]]     
        return {"id_synonyms": self.get_identity_synonyms(), "synonyms": synonyms,
                "soft_nonsynonyms": soft_nonsynonyms, "hard_nonsynonyms": hard_nonsynonyms}
    
    @staticmethod
    def save_semantic_pairs(semantic_data, save_file):
        """Save all the semantic sentence pairs (Note: the output column "label" refers to the ground truth)
        """
        pd.DataFrame(semantic_data, columns=["sentence1", "sentence2", "label"]).to_csv(save_file, sep='\t', index=False)
    
    def negative_sample_check(self, label1, label2):
        """The negative label pair (l1, l2) must satisfy the following conditions:
           1.  l1 != l2; 
           2. (l1, l2) or (l2, l1) not a synonym; 
        """
        # conduct the negative sample check
        if label1 == label2: return False
        if label2 in self.corpus[label1]["synonyms"]: return False
        if label1 in self.corpus[label2]["synonyms"]: return False
        return True

    def save_corpus(self, corpus_file):
        with open(corpus_file, "w") as f:
            json.dump(self.corpus, f, indent=4, separators=(',', ': '), sort_keys=True)
            
    def load_corpus(self, corpus_file):
        with open(corpus_file, "r") as f:
            self.corpus = json.load(f)
        self.corpus_info = self.corpus[" corpus_info "]
        self.corpus_type = self.corpus_info["type"]
        self.from_saved = True
