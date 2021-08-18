"""
Mapping Extension class
"""

from typing import List, Tuple, Dict, Union

import bertmap
from bertmap.onto import OntoBox
from owlready2.entity import ThingClass
from pandas.core.frame import DataFrame
import pandas as pd
from itertools import product
from copy import deepcopy
import time


class OntoExtend:

    na_vals = bertmap.na_vals

    def __init__(
        self,
        src_ob: OntoBox,
        tgt_ob: OntoBox,
        mapping_file: str,
        extend_threshold: float,
    ):

        self.src_ob = src_ob
        self.tgt_ob = tgt_ob
        self.threshold = extend_threshold
        self.raw_mappings = self.read_mappings_to_dict(mapping_file, extend_threshold)
        self.frontier = deepcopy(self.raw_mappings)  # the frontier of mapping expansion
        self.expansion = dict()

    def extend_mappings(self, max_iter: int = 1):
        start_time = time.time()
        num_iter = 0
        while self.frontier and num_iter < max_iter:
            count = 0
            new_expansion = dict()
            for mapping in self.frontier.keys():
                src_iri, tgt_iri = mapping.split("\t")
                print(f"[Time: {round(time.time() - start_time)}][Map {count}]: {src_iri} -> {tgt_iri}")
                sup_maps, sub_maps = self.one_hob_extend(src_iri, tgt_iri)
                # merging dictionary is possible because the duplicates have been removed
                new_expansion = {**new_expansion, **sup_maps, **sub_maps}
                print(f"\t[Iteration {num_iter}]: Extend {len(new_expansion)} new mappings")
                count += 1
            num_iter += 1
            self.frontier = deepcopy(new_expansion)
            self.expansion = {**self.expansion, **new_expansion}
            print(f"[Expansion]: total={len(self.expansion)}")

    def one_hob_extend(self, src_iri: str, tgt_iri: str) -> Tuple[Dict, Dict]:
        """1-hop mapping extension, the assumption is given a highly confident mapping,
        the corresponding classes' parents and children are likely to be matched.

        Args:
            src_iri (str): source class iri
            tgt_iri (str): target class iri
        """
        src_class = self.iri2class(src_iri, flag="SRC")
        tgt_class = self.iri2class(tgt_iri, flag="TGT")

        # detect parents
        src_parents = OntoBox.super_classes(src_class)
        tgt_parents = OntoBox.super_classes(tgt_class)
        sup_maps = self.batch_compute_mapping(src_parents, tgt_parents, "Parents")

        # detect children
        src_children = list(src_class.subclasses())
        tgt_children = list(tgt_class.subclasses())
        sub_maps = self.batch_compute_mapping(src_children, tgt_children, "Children")

        return sup_maps, sub_maps

    def batch_compute_mapping(
        self, src_classes: List[ThingClass], tgt_classes: List[ThingClass], flag: str
    ) -> Dict:
        mappings = dict()
        discarded_mappings = dict()
        seen_mappings = dict()
        for src, tgt in list(product(src_classes, tgt_classes)):
            mapping_str, value = self.compute_mapping(src, tgt)  # ("src_iri\ttgt_iri", value)
            if value >= self.threshold:
                # ensure the mapping is not previously predicted
                mappings[mapping_str] = value
            elif value < self.threshold and value >= 0.0:
                discarded_mappings[mapping_str] = value
            if value == -1.0:
                seen_mappings[mapping_str] = value
        print(
            f"\t[{flag}] found mappings: valid={len(mappings)}, seen={len(seen_mappings)}, discarded={len(discarded_mappings)}"
        )
        # print(discarded_mappings)
        return mappings

    def compute_mapping(
        self, src_class: ThingClass, tgt_class: ThingClass
    ) -> Tuple[str, str, float]:
        """compute the mapping score between src-tgt classes
        IMPORTANT: return a invalid mapping when existed in previously predicted set
        """
        raise NotImplementedError

    def iri2class(self, iri: str, flag: str = "SRC") -> ThingClass:
        """search for the ThingClass object of corresponding iri"""
        assert flag == "SRC" or flag == "TGT"
        ob = self.src_ob if flag == "SRC" else self.tgt_ob
        full_iri = ob.onto_text.expand_entity_iri(iri)
        return ob.onto.search(iri=full_iri)[0]

    @classmethod
    def read_mappings_to_dict(
        cls, mapping_file: Union[str, DataFrame], threshold: float = 0.0
    ) -> Dict:
        """read unique mappings from tsv file or pandas.DataFrame, notice that for duplicated
        mappings, the mapping value is assumed to be consistent.
        """
        if type(mapping_file) is DataFrame:
            _df = mapping_file
        else:
            _df = pd.read_csv(mapping_file, sep="\t", na_values=cls.na_vals, keep_default_na=False)
        mapping_dict = dict()
        for i in range(len(_df)):
            if _df.iloc[i][-1] >= threshold:
                mapping_string = "\t".join(_df.iloc[i][:-1])
                mapping_dict[mapping_string] = _df.iloc[i][-1]
        print(f"read {len(mapping_dict)} mappings with threshold >= {threshold}.")
        return mapping_dict
