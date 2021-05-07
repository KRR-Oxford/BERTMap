"""
Ontology Evaluator class for evaluating the cross-ontology mappings computed by OA models.
"""
import pandas as pd
from bertmap.utils import uniqify
from pandas.core.frame import DataFrame
from bertmap.onto import OntoText
from typing import Optional, Union, List


class OntoEvaluator:
    
    na_vals = pd.io.parsers.STR_NA_VALUES.difference({'NULL', 'null', 'n/a'})
    namespaces = OntoText.namespaces

    def __init__(self, 
                 pre_tsv: Union[str, DataFrame], 
                 ref_tsv: Union[str, DataFrame], 
                 ref_ignored_tsv: Optional[Union[str, DataFrame]]=None, 
                 threshold: float=0.0):
        
        # filter the prediction mappings according to similarity scores
        self.pre = self.read_mappings(pre_tsv, threshold=threshold) 
        
        # reference mappings and illegal mappings to be ignored
        self.ref = self.read_mappings(ref_tsv)
        self.ref_ignored = self.read_mappings(ref_ignored_tsv) if ref_ignored_tsv else None
        
        # compute Precision, Recall and Macro-F1
        self.P = self.precision()
        self.R = self.recall()
        try: self.F1 = self.f1()
        except: self.F1 = "Undefined"
        
    def precision(self) -> float:
        """
        % of predictions are correct:
            P = TP / (TP + FP)
        """
        tp = 0  # true positive
        fp = 0  # false positive
        self.num_ignored = 0
        for p_map in self.pre:
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
            if r_map in self.pre:
                tp += 1
            else:
                fn += 1
        return tp / (tp + fn)

    def f1(self) -> float:
        return 2 * self.P * self.R / (self.P + self.R)
    
    @classmethod
    def read_mappings(cls, 
                      mapping_file: Union[str, DataFrame], 
                      threshold: float=0.0) -> List[str]:
        """read unique mappings from tsv file or pandas.DataFrame
        """
        if type(mapping_file) is DataFrame: _df = mapping_file
        else: _df = pd.read_csv(mapping_file, sep="\t", na_values=cls.na_vals, keep_default_na=False)
        print(threshold)
        mappings = ["\t".join(_df.iloc[i][:-1]) for i in range(len(_df)) if _df.iloc[i][-1] >= threshold]
        return uniqify(mappings)
