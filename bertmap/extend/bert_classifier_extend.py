"""
Mapping Extension class
"""
from bertmap.onto import OntoBox
from bertmap.extend import OntoExtend
from bertmap.map import BERTClassifierMapping
import torch
from owlready2.entity import ThingClass
from itertools import product


class BERTClassifierExtend(OntoExtend):
    def __init__(
        self,
        src_ob: OntoBox,
        tgt_ob: OntoBox,
        mapping_file: str,
        extend_threshold: float,
        bert_checkpoint: str = "some checkpoint",
        tokenizer_path: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_length: int = 128,
        string_match: bool = True,
        device_num: int = 0,
    ):
        super().__init__(src_ob, tgt_ob, mapping_file, extend_threshold)
        self.string_match = string_match
        self.bert_classifier = BERTClassifierMapping(
            src_ob=src_ob,
            tgt_ob=tgt_ob,
            bert_checkpoint=bert_checkpoint,
            tokenizer_path=tokenizer_path,
            max_length=max_length,
            device_num=device_num,
        )

    def compute_mapping(self, src_class: ThingClass, tgt_class: ThingClass):
        """compute the mapping score between src-tgt classes"""

        # retrieve labels from src-tgt classes
        src_class_iri = self.src_ob.onto_text.abbr_entity_iri(src_class.iri)
        src_labels = self.src_ob.onto_text.texts[src_class_iri]["label"]
        tgt_class_iri = self.tgt_ob.onto_text.abbr_entity_iri(tgt_class.iri)
        tgt_labels = self.tgt_ob.onto_text.texts[tgt_class_iri]["label"]
        label_pairs = list(product(src_labels, tgt_labels))
        mapping_str = f"{src_class_iri}\t{tgt_class_iri}"

        # check if the mapping has been explored before
        if mapping_str in self.raw_mappings.keys() or mapping_str in self.expansion.keys():
            # score of -1 means the mapping should be discarded
            return mapping_str, -1.0

        # string-matching module before BERTMap
        if self.string_match:
            for pair in label_pairs:
                if pair[0] == pair[1]:
                    return mapping_str, 1.0

        # apply BERT classifier
        with torch.no_grad():
            model_inputs_dict = self.bert_classifier.tokenize(label_pairs)
            # assign everything to the current device
            for k in model_inputs_dict.keys():
                model_inputs_dict[k] = model_inputs_dict[k].to(self.bert_classifier.device)
            batch_scores = self.bert_classifier.classifier(
                model_inputs_dict
            )  # torch.Size([num_batch_label_pairs]), singleton
            map_score = self.bert_classifier.batch_pooling(
                batch_scores, [len(batch_scores)]
            )  # torch.Size([1]), singleton
            map_score = map_score.cpu().detach().numpy()

        return mapping_str, map_score[0]
