from onto_align.utils import read_mappings, read_onto_uris


class Evaluator:

    def __init__(self, pre, ref):
        src_onto, tgt_onto = read_onto_uris(ref)
        print("----- Load Predicted Mappings -----")
        self.pre_legal, _ = read_mappings(pre, src_onto, tgt_onto)
        print("----- Load Reference Mappings -----")
        self.ref_legal, self.ref_illegal = read_mappings(ref, src_onto, tgt_onto)
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
        unsat = 0
        for p_map in self.pre_legal:
            # ignore the "?" mappings
            if p_map in self.ref_illegal:
                unsat += 1
                continue
            # compute tp and fp non-illegal mappings
            if p_map in self.ref_legal:
                tp += 1
            else:
                fp += 1
        print("#Unsat.:", unsat)
        return tp / (tp + fp)

    def recall(self):
        """
        % of correct retrieved
            R = TP / (TP + FN)
        """
        tp = 0  # true positive
        fn = 0  # false negative
        for r_map in self.ref_legal:
            if r_map in self.pre_legal:
                tp += 1
            else:
                fn += 1
        return tp / (tp + fn)

    def f1(self):
        return 2 * self.P * self.R / (self.P + self.R)
