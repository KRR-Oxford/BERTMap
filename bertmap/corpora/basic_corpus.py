"""
Ontology Corpus superclass, it requires implementation of how to create sub-corpora.
"""

import json
import pandas as pd

class OntologyCorpus:
    
    def __init__(self, *config_args, onto_name=None, corpus_path=None):
        self.onto_name = onto_name
        if not corpus_path:
            self.init_config(*config_args)
            self.create_corpus()
        else:
            # load corpus from local storage
            self.load_corpus(save_dir=corpus_path)
        assert self.corpus_dict is not None
        
    def init_config(self, *config_args):
        raise NotImplementedError
        
    def create_corpus(self):
        raise NotImplementedError

    @staticmethod
    def term_dict():
        # some corpus might not define the hard nonsynonyms so that they will be empty
        return {"synonyms": [], "soft_nonsynonyms": [], "hard_nonsynonyms": []}

    def id_synonyms(self):
        labels = list(self.corpus_dict.keys())[1:]  
        return list(zip(labels, labels, [1]*len(labels)))
    
    def extract_label_pairs(self):
        synonyms = []
        soft_nonsynonyms = []
        hard_nonsynonyms = []
        for term, semantic_dict in self.corpus_dict.items():
            if term == " corpus_info ":
                continue
            synonyms += [(term, s, 1) for s in semantic_dict["synonyms"]]
            soft_nonsynonyms += [(term, ns, 0) for ns in semantic_dict["soft_nonsynonyms"]]
            hard_nonsynonyms += [(term, ns, 0) for ns in semantic_dict["hard_nonsynonyms"]]  
            
        return {"id_synonyms": self.id_synonyms(), "synonyms": synonyms,
                "soft_nonsynonyms": soft_nonsynonyms, "hard_nonsynonyms": hard_nonsynonyms}
    
    @staticmethod
    def backward_label_pairs(label_pairs):
        # (label1, label2, whether_or_not_synonymous)
        return [(y, x, s) for x, y, s in label_pairs]
    
    @staticmethod
    def save_labels(label_data, save_dir_tsv):
        pd.DataFrame(label_data, columns=["Label1", "Label2", "Synonymous"]).to_csv(save_dir_tsv, sep='\t', index=False)
    
    def negative_sample_check(self, label1, label2):
        """The negative label pair (l1, l2) must satisfy the following conditions:
           1.  l1 not = l2; 
           2. (l1, l2) or (l2, l1) not a synonym; 
        """
        # edge case to prevent when label 1 or 2 has not been added into the dict 
        if len(self.corpus_dict[label1]) == 0 or len(self.corpus_dict[label2]) == 0:
            return True
        # conduct the negative sample check
        equiv_cond = label1 == label2
        partition_cond = label2 in self.corpus_dict[label1]["synonyms"] or \
            label1 in self.corpus_dict[label2]["synonyms"]
        if equiv_cond or partition_cond:
            return False
        return True

    def save_corpus(self, save_dir):
        save_file = save_dir + f"/{self.onto_name}.{self.corpus_type}.json"
        with open(save_file, "w") as f:
            json.dump(self.corpus_dict, f, indent=4, separators=(',', ': '), sort_keys=True)
            
    def load_corpus(self, save_dir):
        save_file = save_dir + f"/{self.onto_name}.{self.corpus_type}.json"
        with open(save_file, "r") as f:
            self.corpus_dict = json.load(f)
        self.report(self.corpus_dict)
        
    @staticmethod
    def report(corpus_dict):
        print("--------------- Corpus Summary --------------")
        corpus_info = corpus_dict[" corpus_info "]
        for k, v in corpus_info.items():
            print(f"{k}: {v}")
        print("---------------------------------------------")
