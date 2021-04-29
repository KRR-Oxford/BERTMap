"""
Ontology Evaluator class for evaluating the cross-ontology mappings computed by OA models.
"""
from bertmap.utils.oaei_utils import read_tsv_mappings


class OntoEvaluator:

    def __init__(self, pre_tsv, ref_tsv, ref_ignored_tsv=None, threshold=0.0):
        
        # filter the prediction mappings according to similarity scores
        self.pre = read_tsv_mappings(pre_tsv, threshold=threshold) 
        
        # reference mappings and illegal mappings to be ignored
        self.ref = read_tsv_mappings(ref_tsv)
        self.ref_ignored = read_tsv_mappings(ref_ignored_tsv) if ref_ignored_tsv else None
        
        # compute Precision, Recall and Macro-F1
        self.P = self.precision()
        self.R = self.recall()
        try: self.F1 = self.f1()
        except: self.F1 = "Undefined"
        
    def precision(self):
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
