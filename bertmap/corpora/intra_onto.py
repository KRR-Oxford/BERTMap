"""
Ontology Corpus class from within the input ontology.
"""


import itertools
import random
from collections import defaultdict
from copy import deepcopy
from typing import Optional

from bertmap.corpora import OntoCorpus
from bertmap.onto import OntoBox
from bertmap.utils import exclude_randrange, uniqify


class IntraOntoCorpus(OntoCorpus):
    def __init__(
        self,
        onto_box: Optional[OntoBox] = None,
        sample_rate: int = 10,
        depth_threshold: Optional[int] = None,
        depth_strategy: Optional[str] = "max",
        corpus_file: Optional[str] = None,
    ):

        self.corpus_type = "intra-onto"
        super().__init__(onto_box, sample_rate, depth_threshold, depth_strategy, corpus_file=corpus_file)

    def config(
        self, onto_box: OntoBox, sample_rate: int, depth_threshold: Optional[int], depth_strategy: Optional[str]
    ):

        self.onto_box = onto_box
        self.corpus = defaultdict(lambda: self.init_semantic_dict())
        self.corpus_info = self.init_corpus_info()
        self.corpus_info["onto"].append(self.onto_box.onto.name)
        self.corpus_info["nonsynonyms"]["soft-raw"] = 0
        self.corpus_info["nonsynonyms"]["hard-raw"] = 0
        self.corpus_info["nonsynonyms"]["hard"] = 0
        self.corpus_info["nonsynonyms"]["soft-back"] = 0
        self.corpus_info["nonsynonyms"]["hard-back"] = 0
        self.violations = []
        self.sample_rate = sample_rate  # rate for sampling the random (soft) nonsynonym
        self.corpus_info["others"]["soft_neg_sample_rate"] = self.sample_rate
        self.depth_threshold = depth_threshold  # classes of depths larger than the threshold
        # will not be used for hard nonsynonyms generation
        self.depth_strategy = depth_strategy  # choice of depth generation function
        if depth_threshold and depth_strategy:
            self.onto_box.create_class2depth(depth_strategy)
            self.class2depth = getattr(self.onto_box, f"class2depth_{depth_strategy}")
            self.corpus_info["others"]["max_depth"] = max(self.class2depth.values())
            self.corpus_info["others"]["depth_threshold"] = self.depth_threshold
            self.corpus_info["others"]["depth_strategy"] = self.depth_strategy

    def create_corpus(self):
        self.intra_onto_synonyms()
        self.intra_onto_soft_nonsynonyms()
        self.intra_onto_hard_nonsynonyms()
        self.corpus_info["synonyms"]["id"] = len(self.corpus)
        self.corpus_info["nonsynonyms"]["removed_violations"] = len(self.violations)
        self.corpus[" corpus_info "] = self.corpus_info

    def intra_onto_synonyms(self):
        """The synonyms include identity, forward and backward as :
             (a_i, a_i); [(a_i, a_{j>i}); (a_i, a_{j<i})
        but we only need to store the *forward* and *backward* because the identity pairs
        can be easily generated from the dictionary data structure.
        """
        for _, text_dict in self.onto_box.onto_text.texts.items():
            labels = text_dict["label"]
            for i in range(len(labels)):
                label = labels[i]
                aliases = deepcopy(self.corpus[label]["synonyms"])
                existed_num = len(aliases)
                # store for and back synonym pairs and remove duplicates
                aliases = uniqify(aliases + labels[:i] + labels[i + 1 :])
                self.corpus_info["synonyms"]["non-id"] += len(aliases) - existed_num
                self.corpus[label]["synonyms"] = aliases

    def intra_onto_soft_nonsynonyms(self):
        """For each label of class C, we sample N negative labels from N random classes that are not C."""
        num_classes = len(self.onto_box.onto_text.class2idx)
        for class_iri, text_dict in self.onto_box.onto_text.texts.items():
            idx = self.onto_box.onto_text.class2idx[class_iri]
            labels = text_dict["label"]
            for label in labels:
                soft_negatives = deepcopy(self.corpus[label]["soft_nonsynonyms"])
                existed_num = len(soft_negatives)
                neg_idxs = [exclude_randrange(0, num_classes, exclude=idx) for _ in range(self.sample_rate)]
                assert len(neg_idxs) == self.sample_rate
                for nidx in neg_idxs:
                    neg_cls_iri = self.onto_box.onto_text.idx2class[nidx]
                    neg_label = random.choice(self.onto_box.onto_text.texts[neg_cls_iri]["label"])
                    if self.negative_sample_check(label, neg_label):
                        soft_negatives.append(neg_label)
                        ######## update the backward nonsynonym #######
                        if not label in self.corpus[neg_label]["soft_nonsynonyms"]:
                            self.corpus[neg_label]["soft_nonsynonyms"].append(label)
                            self.corpus_info["nonsynonyms"]["soft"] += 1
                            self.corpus_info["nonsynonyms"]["soft-back"] += 1
                        ################################################
                    else:
                        self.violations.append([label, neg_label])
                self.corpus_info["nonsynonyms"]["soft-raw"] += len(soft_negatives) - existed_num
                soft_negatives = uniqify(soft_negatives)
                self.corpus_info["nonsynonyms"]["soft"] += len(soft_negatives) - existed_num
                self.corpus[label]["soft_nonsynonyms"] = soft_negatives

    def intra_onto_hard_nonsynonyms(self):
        """Assume the subclasses of a class C at depth <= D (if specified) are disjoint,
        we define the (L, L') as hard nonsynonyms for L from one of the subclasses and L' from the other.
        If D is not specified, then C at any depth will be considered as long as it has multiple sub-classes.
        """
        for cl in self.onto_box.onto.classes():
            # skip if the depth of the class exceeds the threshold
            cl_iri = self.onto_box.onto_text.abbr_entity_iri(cl.iri)
            if self.depth_threshold and self.depth_strategy:
                depth = self.class2depth[cl_iri]
                if depth > self.depth_threshold:
                    continue
            sub_classes = [self.onto_box.onto_text.abbr_entity_iri(scl.iri) for scl in cl.subclasses()]
            # skip if no sibling classes available
            if len(sub_classes) <= 1:
                continue
            # with at least two sibiling classes we can extract hard negatives
            # [[sibiling class 1's labels], [sibling class 2's labels], ...]
            sib_labels = [self.onto_box.onto_text.texts[scl]["label"] for scl in sub_classes]
            # e.g. sibling1: ["a", "b"]; sibling2: ["c"] -> [("a", "c"), ("b", "c")] as forward
            for i in range(len(sib_labels)):
                labels = sib_labels[i]
                neg_labels = list(itertools.chain.from_iterable(sib_labels[i + 1 :]))
                for label in labels:
                    hard_negatives = deepcopy(self.corpus[label]["hard_nonsynonyms"])
                    existed_num = len(hard_negatives)
                    for neg_label in neg_labels:
                        if self.negative_sample_check(label, neg_label):
                            hard_negatives.append(neg_label)
                            ######## update the backward nonsynonym #######
                            if not label in self.corpus[neg_label]["hard_nonsynonyms"]:
                                self.corpus[neg_label]["hard_nonsynonyms"].append(label)
                                self.corpus_info["nonsynonyms"]["hard"] += 1
                                self.corpus_info["nonsynonyms"]["hard-back"] += 1
                            ################################################
                        else:
                            self.violations.append([label, neg_label])
                    self.corpus_info["nonsynonyms"]["hard-raw"] += len(hard_negatives) - existed_num
                    hard_negatives = uniqify(hard_negatives)
                    self.corpus_info["nonsynonyms"]["hard"] += len(hard_negatives) - existed_num
                    self.corpus[label]["hard_nonsynonyms"] = hard_negatives
