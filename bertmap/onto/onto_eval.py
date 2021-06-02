"""Ontology Evaluator class for evaluating the cross-ontology mappings computed by OA models.
"""
from typing import List, Optional, Union

import pandas as pd

import bertmap
from bertmap.utils import uniqify
from pandas.core.frame import DataFrame


class OntoEvaluator:

    na_vals = bertmap.na_vals
    namespaces = bertmap.namespaces

    def __init__(
        self,
        pre_tsv: Union[str, DataFrame],
        ref_tsv: Union[str, DataFrame],
        ref_ignored_tsv: Optional[Union[str, DataFrame]] = None,
        threshold: float = 0.0,
    ):

        # filter the prediction mappings according to similarity scores
        self.pre = self.read_mappings(pre_tsv, threshold=threshold)

        # reference mappings and illegal mappings to be ignored
        self.ref = self.read_mappings(ref_tsv)
        if ref_ignored_tsv is None:
            self.ref_ignored = None
        else:
            self.ref_ignored = self.read_mappings(ref_ignored_tsv)

        # compute Precision, Recall and Macro-F1
        self.num_ignored = 0
        self.id_maps = []  # mapping of same iris should be ignored
        try:
            self.P = self.precision()
            self.R = self.recall()
            self.F1 = self.f1()
        except:
            self.P = -1
            self.R = -1
            self.F1 = -1

    def precision(self) -> float:
        """
        % of predictions are correct:
            P = TP / (TP + FP)
        """
        tp = 0  # true positive
        fp = 0  # false positive
        self.num_ignored = 0
        self.id_maps = []
        for p_map in self.pre:
            # mapping of same iris should be ignored
            if p_map.split("\t")[0] == p_map.split("\t")[1]:
                self.id_maps.append(p_map)
                self.num_ignored += 1
                continue
            # ignore the "?" mappings where available
            if self.ref_ignored and p_map in self.ref_ignored:
                self.num_ignored += 1
                continue
            # compute tp and fp non-illegal mappings
            if p_map in self.ref:
                tp += 1
            else:
                fp += 1
        return tp / (tp + fp)

    def recall(self) -> float:
        """
        % of correct retrieved
            R = TP / (TP + FN)
        """
        tp = 0  # true positive
        fn = 0  # false negative
        for r_map in self.ref:
            # ignore the "?" mappings where available
            if self.ref_ignored and r_map in self.ref_ignored:
                self.num_ignored += 1
                continue
            if r_map in self.pre:
                tp += 1
            else:
                fn += 1
        return tp / (tp + fn)

    def f1(self) -> float:
        return 2 * self.P * self.R / (self.P + self.R)

    @classmethod
    def read_mappings(
        cls, mapping_file: Union[str, DataFrame], threshold: float = 0.0
    ) -> List[str]:
        """read unique mappings from tsv file or pandas.DataFrame"""
        if type(mapping_file) is DataFrame:
            _df = mapping_file
        else:
            _df = pd.read_csv(mapping_file, sep="\t", na_values=cls.na_vals, keep_default_na=False)
        mappings = [
            "\t".join(_df.iloc[i][:-1]) for i in range(len(_df)) if _df.iloc[i][-1] >= threshold
        ]
        return uniqify(mappings)
