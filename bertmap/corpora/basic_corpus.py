"""
Ontology Corpus superclass, it requires implementation of how to create sub-corpora.
"""

import json

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
        labels = list(self.corpus_dict.keys())
        return list(zip(labels, labels))
    
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
