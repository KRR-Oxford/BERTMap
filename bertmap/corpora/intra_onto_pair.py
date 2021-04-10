from bertmap.corpora import OntologyCorpus
from bertmap.corpora import IntraOntoCorpus
from bertmap.utils import uniqify
from copy import deepcopy


class IntraOntoCorpusPair(OntologyCorpus):
    
    def __init__(self, task_name, src_intra_onto_corpus=None, tgt_intra_onto_corpus=None, corpus_path=None):
        super().__init__(task_name, src_intra_onto_corpus, tgt_intra_onto_corpus, corpus_path=corpus_path)
        
    def init_config(self, task_name, src_intra_onto_corpus: IntraOntoCorpus, tgt_intra_onto_corpus: IntraOntoCorpus):
        self.corpus_dict = deepcopy(src_intra_onto_corpus.corpus_dict)
        self.to_add_corpus_dict = deepcopy(tgt_intra_onto_corpus.corpus_dict)
        print("Merging the following Source and Target Ontologies ...")
        self.report(self.corpus_dict)
        self.report(self.to_add_corpus_dict)
        self.onto_name = task_name
        self.corpus_type = "intra-onto-pair"
        
    def create_corpus(self):
        self.corpus_dict[" corpus_info "]["num_violated"] += self.to_add_corpus_dict[" corpus_info "]["num_violated"]
        self.update_synonyms()
        self.update_nonsynonyms("soft")
        self.update_nonsynonyms("hard")
        self.corpus_dict[" corpus_info "]["corpus_type"] =  "Intra-ontology Corpus (Pair)"
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
