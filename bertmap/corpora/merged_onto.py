"""
Ontology Corpus class that takes as input a base corpus and an additional corpus for update.
"""


import random
from copy import deepcopy
from typing import Optional

from bertmap.corpora import OntoCorpus
from bertmap.utils import uniqify
from sklearn.model_selection import train_test_split


class MergedOntoCorpus(OntoCorpus):
    def __init__(
        self,
        base_onto_corpus: Optional[OntoCorpus] = None,
        add_onto_corpus: Optional[OntoCorpus] = None,
        corpus_file: Optional[str] = None,
    ):
        self.corpus_type = "merged-onto"
        super().__init__(base_onto_corpus, add_onto_corpus, corpus_file=corpus_file)

    def __repr__(self):
        if self.from_saved:
            return super().__repr__()
        report = super().__repr__().replace("\n</OntologyCorpus>", "")
        report += "\n\t<!--the following two corpora were merged-->"
        report += f"\n{str(self.base_repr)}".replace("\n", "\n\t")
        report += f"\n{str(self.add_repr)}".replace("\n", "\n\t")
        report += "\n</OntologyCorpus>"
        return report

    def config(self, base_onto_corpus: OntoCorpus, add_onto_corpus: OntoCorpus):
        self.corpus = deepcopy(base_onto_corpus.corpus)
        self.corpus_info = self.corpus[" corpus_info "]
        self.corpus_info["others"] = dict()
        self.add_corpus = deepcopy(add_onto_corpus.corpus)
        self.add_corpus_info = self.add_corpus[" corpus_info "]
        self.corpus_info["nonsynonyms"]["removed_violations"] += self.add_corpus_info["nonsynonyms"][
            "removed_violations"
        ]
        self.corpus_info["onto"] += self.add_corpus_info["onto"]
        del self.corpus_info["nonsynonyms"]["soft-back"]
        del self.corpus_info["nonsynonyms"]["hard-back"]
        del self.corpus_info["nonsynonyms"]["soft-raw"]
        del self.corpus_info["nonsynonyms"]["hard-raw"]
        self.violations = []
        self.base_repr = str(base_onto_corpus)
        self.add_repr = str(add_onto_corpus)

    def create_corpus(self):
        self.update_synonyms()
        self.update_nonsynonyms("soft")
        self.update_nonsynonyms("hard")
        self.corpus_info["synonyms"]["id"] = len(self.corpus)
        self.corpus_info["nonsynonyms"]["removed_violations"] += len(self.violations)
        self.corpus[" corpus_info "] = self.corpus_info
        print("finishing merging (print the merged-corpus to see updated details) ...")

    def update_synonyms(self):
        for add_label, add_dict in self.add_corpus.items():
            if add_label == " corpus_info ":
                continue
            # extract the existed term dict or initialize an empty one
            aliases = deepcopy(self.corpus[add_label]["synonyms"])
            existed_num = len(aliases)
            aliases = uniqify(aliases + add_dict["synonyms"])
            self.corpus_info["synonyms"]["non-id"] += len(aliases) - existed_num
            self.corpus[add_label]["synonyms"] = aliases

    def update_nonsynonyms(self, neg_str: str = "soft"):
        assert neg_str == "soft" or neg_str == "hard"
        for add_label, add_dict in self.add_corpus.items():
            if add_label == " corpus_info ":
                continue
            # extract the existed term dict or initialize an empty one
            negatives = deepcopy(self.corpus[add_label][f"{neg_str}_nonsynonyms"])
            existed_num = len(negatives)
            add_negatives = add_dict[f"{neg_str}_nonsynonyms"]
            for neg_label in add_negatives:
                # the negative sample must not be existed in the updated synonym set
                if self.negative_sample_check(add_label, neg_label):
                    negatives.append(neg_label)
                else:
                    self.violations.append([add_label, neg_label])
            negatives = uniqify(negatives)
            self.corpus_info["nonsynonyms"][neg_str] += len(negatives) - existed_num
            self.corpus[add_label][f"{neg_str}_nonsynonyms"] = negatives

    def train_val_split(self, val_ratio: float = 0.2, soft_neg_rate: int = 1, hard_neg_rate: int = 1):

        semantic_pairs = self.extract_semantic_pairs()

        # label data without identity synonyms
        synonyms = semantic_pairs["synonyms"]  # for-back synonyms
        # check if the input negative rates are applicable
        if len(semantic_pairs["soft_nonsynonyms"]) < soft_neg_rate * len(synonyms):
            print(
                "# soft-nonsynonyms in the current corpus is not enough, \
                reduce the soft-negative rate or re-create the corpus with higher sample rate."
            )
            return None
        if len(semantic_pairs["hard_nonsynonyms"]) < hard_neg_rate * len(synonyms):
            print(
                "# hard-nonsynonyms in the current corpus is not enough, \
            reduce the soft-negative rate or re-create the corpus with higher sample rate."
            )
            return None
        # for each synonym, sample {soft_neg_rate} soft and {hard_neg_rate} hard negatives
        soft_nonsynonyms = random.sample(semantic_pairs["soft_nonsynonyms"], soft_neg_rate * len(synonyms))
        hard_nonsynonyms = random.sample(semantic_pairs["hard_nonsynonyms"], hard_neg_rate * len(synonyms))
        # form the label data
        label_data = synonyms + soft_nonsynonyms + hard_nonsynonyms
        label_data = uniqify(label_data)
        random.shuffle(label_data)

        # label data with identity synonyms
        id_synonyms = semantic_pairs["id_synonyms"]
        # sample negatives according to the size of identity synonyms
        soft_nonsynonyms_for_ids = random.sample(semantic_pairs["soft_nonsynonyms"], soft_neg_rate * len(id_synonyms))
        hard_nonsynonyms_for_ids = random.sample(semantic_pairs["hard_nonsynonyms"], hard_neg_rate * len(id_synonyms))
        label_data_for_ids = id_synonyms + soft_nonsynonyms_for_ids + hard_nonsynonyms_for_ids
        label_data_for_ids = uniqify(label_data_for_ids)
        random.shuffle(label_data_for_ids)

        if val_ratio == 0.0:
            return label_data, label_data_for_ids
        else:
            train, val = train_test_split(label_data, test_size=val_ratio)
            train_ids, val_ids = train_test_split(label_data_for_ids, test_size=val_ratio)
            return train, val, train_ids, val_ids
