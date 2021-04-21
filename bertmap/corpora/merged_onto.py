"""
Ontology Corpus class that takes as input a base corpus and an additional corpus for update.
"""


from bertmap.corpora import OntologyCorpus
from bertmap.utils import uniqify
from copy import deepcopy
from sklearn.model_selection import train_test_split
import random


class MergedOntoCorpus(OntologyCorpus):
    
    def __init__(self, onto_name, base_onto_corpus=None, to_add_onto_corpus=None, corpus_path=None):
        self.corpus_type = "merged-onto"
        super().__init__(base_onto_corpus, to_add_onto_corpus, onto_name=onto_name, corpus_path=corpus_path)
        
    def init_config(self, base_onto_corpus: OntologyCorpus, to_add_onto_corpus: OntologyCorpus):
        self.corpus_dict = deepcopy(base_onto_corpus.corpus_dict)
        self.to_add_corpus_dict = deepcopy(to_add_onto_corpus.corpus_dict)
        print("Merging the following Source and Target Ontologies ...")
        self.report(self.corpus_dict)
        self.report(self.to_add_corpus_dict)
        
    def create_corpus(self):
        self.corpus_dict[" corpus_info "]["num_violated"] += self.to_add_corpus_dict[" corpus_info "]["num_violated"]
        self.update_synonyms()
        self.update_nonsynonyms("soft")
        self.update_nonsynonyms("hard")
        self.corpus_dict[" corpus_info "]["corpus_type"] =  "Merged Ontology Corpus"
        self.corpus_dict[" corpus_info "]["corpus_onto"] = self.onto_name
        self.corpus_dict[" corpus_info "]["id_synonyms"] = len(self.corpus_dict)
        print("Updated Corpora Infomation ...")
        self.report(self.corpus_dict)

    def update_synonyms(self):
        corpus_info = self.corpus_dict[" corpus_info "]
        for to_add_term, to_add_term_dict in self.to_add_corpus_dict.items():
            if to_add_term == " corpus_info ":
                continue
            # extract the existed term dict or initialize an empty one
            term_dict = self.corpus_dict[to_add_term]
            ###### For updating the synonyms ######
            synonym_list = deepcopy(term_dict["synonyms"])
            existed_num = len(synonym_list)
            to_add_synonym_list = deepcopy(to_add_term_dict["synonyms"])
            synonym_list = uniqify(synonym_list + to_add_synonym_list)
            corpus_info["synonyms"] += len(synonym_list) - existed_num
            ##### Update the dictionary #####
            term_dict["synonyms"] = synonym_list
            self.corpus_dict[to_add_term] = term_dict
        # update the corpus_info
        self.corpus_dict[" corpus_info "] = corpus_info
            
    def update_nonsynonyms(self, flag="soft"):
        assert flag == "soft" or flag == "hard"
        nonsynonym_string = flag + "_nonsynonyms"
        corpus_info = self.corpus_dict[" corpus_info "]
        for to_add_term, to_add_term_dict in self.to_add_corpus_dict.items():
            if to_add_term == " corpus_info ":
                continue
            # extract the existed term dict or initialize an empty one
            term_dict = self.corpus_dict[to_add_term]
            ###### For updating the synonyms ######
            nonsynonym_list = deepcopy(term_dict[nonsynonym_string])
            existed_num = len(nonsynonym_list)
            tgt_nonsynonym_list = to_add_term_dict[nonsynonym_string] 
            for tgt_nonsynonym in tgt_nonsynonym_list:
                # the negative sample must not be existed in the updated synonym set
                if self.negative_sample_check(to_add_term, tgt_nonsynonym):
                    nonsynonym_list.append(tgt_nonsynonym)
                else:
                    corpus_info["num_violated"] += 1
            nonsynonym_list = uniqify(nonsynonym_list)
            corpus_info[nonsynonym_string] += len(nonsynonym_list) - existed_num
            ##### Update the dictionary #####
            term_dict[nonsynonym_string] = nonsynonym_list
            self.corpus_dict[to_add_term] = term_dict
        # update the corpus_info
        self.corpus_dict[" corpus_info "] = corpus_info
        
    def train_val_split(self, train_ratio=0.8, val_ratio=0.2, backward=False, identity=False, only_train=False):
        assert train_ratio + val_ratio == 1.0
        onto_labels = self.extract_label_pairs()
        # extract cross-ontology labels
        synonyms = onto_labels["synonyms"]  # forward synonyms
         
        # for each synonym, sample a soft and a hard negative
        # in case there are too many synonyms (especially after adding the identity synonyms)
        # we use all the nonsynonyms available
        soft_nonsynonyms = random.sample(onto_labels["soft_nonsynonyms"], len(synonyms)) \
            if len(synonyms) < len(onto_labels["soft_nonsynonyms"]) else onto_labels["soft_nonsynonyms"]
        hard_nonsynonyms = random.sample(onto_labels["hard_nonsynonyms"], len(synonyms)) \
            if len(synonyms) < len(onto_labels["hard_nonsynonyms"]) else onto_labels["hard_nonsynonyms"]
        
        # add all the backward label pairs
        if backward:
            synonyms += self.backward_label_pairs(synonyms)
            soft_nonsynonyms += self.backward_label_pairs(soft_nonsynonyms)
            hard_nonsynonyms += self.backward_label_pairs(hard_nonsynonyms)
            
        if identity:
            synonyms += onto_labels["id_synonyms"]  # add identity synonyms
            
        # form the label data
        label_data = synonyms + soft_nonsynonyms + hard_nonsynonyms
        label_data = uniqify(label_data)
        random.shuffle(label_data)
        if only_train:
            return label_data
        else:
            train, val = train_test_split(label_data, test_size=val_ratio)
            return train, val

