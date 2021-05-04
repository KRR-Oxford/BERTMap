"""
Ontology Corpus class from known/seed cross-ontology mappings.
"""

from bertmap.onto import OntoBox
from bertmap.corpora import OntoCorpus
from bertmap.utils.oaei_utils import read_tsv_mappings
from bertmap.utils import uniqify, exclude_randrange
from collections import defaultdict
from sklearn.model_selection import train_test_split
from copy import deepcopy
from typing import Union, Optional, List
import random

class CrossOntoCorpus(OntoCorpus):
    
    def __init__(self, 
                 src_onto_box: Optional[OntoBox]=None, 
                 tgt_onto_box: Optional[OntoBox]=None,
                 src2tgt_mappings_file: Optional[Union[str, List[str]]]=None,
                 sample_rate: int=10, 
                 corpus_file: Optional[str]=None):
        self.corpus_type = "cross-onto"
        super().__init__(src_onto_box, tgt_onto_box, src2tgt_mappings_file, 
                         sample_rate, corpus_file=corpus_file)
        
    @classmethod
    def spliting_corpus(cls, 
                        src_onto_box: OntoBox, 
                        tgt_onto_box: OntoBox, 
                        src2tgt_mappings_file: str, 
                        train_ratio: float=0.2,
                        val_ratio: float=0.1,
                        test_ratio: float=0.7,
                        sample_rate: int=10):
        """return three cross-onto corpora by splitting the reference mappings according to the specified ratios
           this refers to the semi-supervised setting where a small portion of high-confidence mappings is provided
        """
        assert train_ratio + val_ratio + test_ratio == 1.0
        train_maps, val_test_maps = train_test_split(read_tsv_mappings(src2tgt_mappings_file), test_size=1-train_ratio)
        val_maps, test_maps = train_test_split(val_test_maps, test_size=test_ratio/(val_ratio + test_ratio))
        kwargs = {"src_onto_box": src_onto_box, "tgt_onto_box": tgt_onto_box, "sample_rate": sample_rate}
        return \
            cls(src2tgt_mappings_file=train_maps, **kwargs), \
            cls(src2tgt_mappings_file=val_maps, **kwargs), \
            cls(src2tgt_mappings_file=test_maps, **kwargs)
        
        
    def config(self, 
               src_onto_box: OntoBox, 
               tgt_onto_box: OntoBox, 
               src2tgt_mappings_file: Optional[Union[str, List[str]]], 
               sample_rate: int):
        self.src_ob = src_onto_box
        self.tgt_ob = tgt_onto_box
        # separate the mappings if certain portion is assumed as known
        if type(src2tgt_mappings_file) is str: self.maps = read_tsv_mappings(src2tgt_mappings_file)
        else: self.maps = src2tgt_mappings_file
        # initialize corpus
        self.corpus = defaultdict(lambda:self.semantic_dict())
        self.corpus_info = self.corpus_info()
        self.sample_rate = sample_rate  # rate for sampling the random (soft) nonsynonym
        self.violations = []
        self.corpus_info["onto"].append(self.src_ob.onto.name)
        self.corpus_info["onto"].append(self.tgt_ob.onto.name)
        self.corpus_info["others"]["soft_neg_sample_rate"] = self.sample_rate
        self.corpus_info["nonsynonyms"]["soft-back"] = 0
        self.corpus_info["nonsynonyms"]["soft-raw"] = 0
            
    def create_corpus(self):
        self.cross_onto_synonyms()
        self.cross_onto_nonsynonyms()
        self.corpus_info["nonsynonyms"]["removed_violations"] += len(self.violations)
        self.corpus[" corpus_info "] = self.corpus_info
        
    def cross_onto_synonyms(self):
        for i in range(len(self.maps)):
            src_cls, tgt_cls = self.maps[i].split("\t")
            src_labels = self.src_ob.onto_text.texts[src_cls]["label"]
            tgt_labels = self.tgt_ob.onto_text.texts[tgt_cls]["label"]
            # fix a src-class label, add tgt-class labels
            for src_label in src_labels:
                aliases = deepcopy(self.corpus[src_label]["synonyms"])
                existed_num = len(aliases)
                aliases = uniqify(aliases + tgt_labels)
                self.corpus_info["synonyms"]["non-id"] += len(aliases) - existed_num
                self.corpus[src_label]["synonyms"] = aliases
            # fix a tgt-class label, add src-class labels
            for tgt_label in tgt_labels:
                aliases = deepcopy(self.corpus[tgt_label]["synonyms"])
                existed_num = len(aliases)
                aliases = uniqify(aliases + src_labels)
                self.corpus_info["synonyms"]["non-id"] += len(aliases) - existed_num
                self.corpus[tgt_label]["synonyms"] = aliases
 
    def cross_onto_nonsynonyms(self):
        """There is no hard nonsynonym available on the cross-ontology level.
           Here the nonsynonyms are extracted from randomly mis-aligned classes.
        """
        for i in range(len(self.maps)):
            src_cls, tgt_cls = self.maps[i].split("\t")
            src_labels = self.src_ob.onto_text.texts[src_cls]["label"]
            tgt_labels = self.tgt_ob.onto_text.texts[tgt_cls]["label"]
            self.negative_sampling(src_labels, i, search_from="TGT")
            self.negative_sampling(tgt_labels, i, search_from="SRC")

    def negative_sampling(self, cls_labels: List[str], map_idx: int, search_from: str="SRC"):
        """The helper function for sampling {sample_rate} negative labels from 
           opposite ontology O' for each class label of the ontology O
        """
        if search_from == "SRC": onto_text = self.src_ob.onto_text
        elif search_from == "TGT": onto_text = self.tgt_ob.onto_text
        else: raise TypeError
        for label in cls_labels:
            neg_idxs = [exclude_randrange(0, len(self.maps), exclude=map_idx) for _ in range(self.sample_rate)]
            assert len(neg_idxs) == self.sample_rate
            soft_negatives = deepcopy(self.corpus[label]["soft_nonsynonyms"])
            existed_num = len(soft_negatives)
            for nidx in neg_idxs:
                src_neg_cls_iri, tgt_neg_cls_iri = self.maps[nidx].split("\t")
                neg_cls_iri = src_neg_cls_iri if search_from == "SRC" else tgt_neg_cls_iri
                neg_class_iri = onto_text.abbr_entity_iri(neg_cls_iri)
                neg_label = random.choice(onto_text.texts[neg_class_iri]["label"])
                if self.negative_sample_check(label, neg_label):
                    soft_negatives.append(neg_label)
                    ######## update the backward nonsynonym #######
                    if not label in self.corpus[neg_label]["soft_nonsynonyms"]:
                        self.corpus[neg_label]["soft_nonsynonyms"].append(label)
                        self.corpus_info["nonsynonyms"]["soft"] += 1
                        self.corpus_info["nonsynonyms"]["soft-back"] += 1
                    ################################################
                else: self.violations.append([label, neg_label])
            self.corpus_info["nonsynonyms"]["soft-raw"] += len(soft_negatives) - existed_num
            soft_negatives = uniqify(soft_negatives)
            self.corpus_info["nonsynonyms"]["soft"] += len(soft_negatives) - existed_num
            self.corpus[label]["soft_nonsynonyms"] = soft_negatives    
            
    def generate_label_data(self, soft_neg_rate: int=2):
        semantic_pairs = self.extract_semantic_pairs()
        synonyms = semantic_pairs["synonyms"]  # for-back synonyms
        # check if the input negative rates are applicable
        if len(semantic_pairs["soft_nonsynonyms"]) < soft_neg_rate * len(synonyms): 
            print("# soft-nonsynonyms in the current corpus is not enough, \
                reduce the soft-negative rate or re-create the corpus with higher sample rate.")
            return None
        # for each synonym, sample {soft_neg_rate} soft negatives
        soft_nonsynonyms = random.sample(semantic_pairs["soft_nonsynonyms"], soft_neg_rate * len(synonyms))
        # form the label data
        label_data = synonyms + soft_nonsynonyms
        label_data = uniqify(label_data)
        random.shuffle(label_data)
        return label_data
