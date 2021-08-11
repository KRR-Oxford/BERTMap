"""
Mapping Extension class
"""

from typing import Union, List, Tuple

from bertmap.onto import OntoBox, OntoEvaluator
from pandas import DataFrame
from owlready2.entity import ThingClass
from itertools import product


class OntoExtend:
    def __init__(
        self,
        src_ob: OntoBox,
        tgt_ob: OntoBox,
        extend_threshold: float,
    ):

        self.src_ob = src_ob
        self.tgt_ob = tgt_ob
        self.threshold = extend_threshold

    def extend_mappings(self, pred_mappings: List[str]):
        expansion = []
        count = 0
        for p_map in pred_mappings:
            src_iri, tgt_iri = p_map.split("\t")
            sup_maps, sub_maps = self.one_hob_extend(src_iri, tgt_iri)
            print(f"[Map {count}]: {src_iri} -> {tgt_iri}")
            print(f"\t# extended mappings in: parents={len(sup_maps)}; children={len(sub_maps)}")
            expansion += sup_maps
            expansion += sub_maps
            count += 1
        print(f"Extend {len(expansion)} mappings.")
        return expansion

    def one_hob_extend(self, src_iri: str, tgt_iri: str) -> Tuple[List, List]:
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
        sup_maps = self.batch_compute_mapping(src_parents, tgt_parents)

        # detect children
        src_children = list(src_class.subclasses())
        tgt_children = list(tgt_class.subclasses())
        sub_maps = self.batch_compute_mapping(src_children, tgt_children)

        return sup_maps, sub_maps

    def batch_compute_mapping(
        self, src_classes: List[ThingClass], tgt_classes: List[ThingClass]) -> List:
        mappings = []
        for src, tgt in list(product(src_classes, tgt_classes)):
            extended_mapping = self.compute_mapping(src, tgt)  # (src_iri, tgt_iri, value)
            if extended_mapping[2] > self.threshold:
                mappings.append(extended_mapping)
        return mappings

    def compute_mapping(
        self, src_class: ThingClass, tgt_class: ThingClass
    ) -> Tuple[str, str, float]:
        """compute the mapping score between src-tgt classes"""
        raise NotImplementedError

    def iri2class(self, iri: str, flag: str = "SRC") -> ThingClass:
        """search for the ThingClass object of corresponding iri"""
        assert flag == "SRC" or flag == "TGT"
        ob = self.src_ob if flag == "SRC" else self.tgt_ob
        full_iri = ob.onto_text.expand_entity_iri(iri)
        return ob.onto.search(iri=full_iri)[0]
