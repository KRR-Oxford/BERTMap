"""
For evaluating the ontology mappings
"""
from onto_align.onto.oaei_utils import read_tsv_mappings


class OntoEvaluator:

    def __init__(self, pre_tsv, ref_tsv, except_tsv=None, threshold=0.0):
        self.pre = read_tsv_mappings(pre_tsv, threshold=threshold)  # filter the prediction mappings according to similarity scores
        self.ref = read_tsv_mappings(ref_tsv)
        self.ref_illegal = read_tsv_mappings(except_tsv) if except_tsv else None
        self.P = self.precision()
        self.R = self.recall()
        self.F1 = self.f1()
        
    def precision(self):
        """
        % of predictions are correct:
            P = TP / (TP + FP)
        """
        tp = 0  # true positive
        fp = 0  # false positive
        num_illegal = 0
        for p_map in self.pre:
            # ignore the "?" mappings where available
            if self.ref_illegal and p_map in self.ref_illegal:
                num_illegal += 1
                continue
            # compute tp and fp non-illegal mappings
            if p_map in self.ref:
                tp += 1
            else:
                fp += 1
        self.num_illegal = num_illegal
        return tp / (tp + fp)

    def recall(self):
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

    def f1(self):
        return 2 * self.P * self.R / (self.P + self.R)
