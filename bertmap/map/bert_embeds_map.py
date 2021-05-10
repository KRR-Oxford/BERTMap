"""Mapping Generation on using Pretrained/Fine-tuned BERT with various pooling strategies and cosine-similarity.
"""

import time
from typing import List, Optional

import torch
from bertmap.bert import BERTStatic
from bertmap.map import OntoMapping
from bertmap.onto import OntoBox
from bertmap.utils import get_device
from sklearn.metrics.pairwise import cosine_similarity


class BERTEmbedsMapping(OntoMapping):
    def __init__(
        self,
        src_ob: OntoBox,
        tgt_ob: OntoBox,
        candidate_limit: Optional[int] = 50,
        save_dir: str = "",
        batch_size: int = 32,
        nbest: int = 1,
        bert_checkpoint: str = "some checkpoint",
        tokenizer_path: str = "emilyalsentzer/Bio_ClinicalBERT",
        string_match: bool = True,
        strategy: str = "mean",
        device_num: int = 0,
    ):

        super().__init__(src_ob, tgt_ob, candidate_limit, save_dir)
        # basic attributes
        self.batch_size = batch_size
        self.nbest = nbest
        self.string_match = string_match
        self.strategy = strategy
        assert self.strategy == "mean" or self.strategy == "cls"

        self.bert = BERTStatic(bert_checkpoint=bert_checkpoint, tokenizer_path=tokenizer_path, with_classifier=False)
        self.device = get_device(device_num=device_num)
        self.bert.model.to(self.device)

    def alignment(self, flag="SRC"):
        self.start_time = time.time()
        print_flag = (
            f"{flag}: {self.src_ob.onto_text.iri_abbr}"
            if flag == "SRC"
            else f"{flag}: {self.tgt_ob.onto_text.iri_abbr}"
        )
        from_ob, to_ob = self.from_to_config(flag=flag)
        i = 0
        for from_class in from_ob.onto.classes():
            from_class_iri = from_ob.onto_text.abbr_entity_iri(from_class.iri)
            from_labels = from_ob.onto_text.texts[from_class_iri]["label"]
            search_space = (
                to_ob.onto_text.text.keys()
                if not self.candidate_limit
                else to_ob.select_candidates(from_labels, self.candidate_limit)
            )
            from_class_idx = from_ob.onto_text.class2idx[from_class_iri]
            assert from_class_idx == i
            i += 1  # to test the order preservation in OntoText dict
            if len(search_space) == 0:
                self.log_print(
                    f"[Time: {round(time.time() - self.start_time)}][{print_flag}][Class-idx: {from_class_idx}] No candidates available for for current entity ..."
                )
                continue
            nbest_results = self.batch_alignment(from_class_iri, from_labels, search_space, flag=flag)
            for to_class_iri, mapping_score in nbest_results:
                if mapping_score <= 0.01:
                    mapping_score = 0.0
                result = (from_class_iri, to_class_iri, mapping_score)
                self.log_print(
                    f"[Time: {round(time.time() - self.start_time)}][{print_flag}][Class-idx: {from_class_idx}][Mapping: {result}]"
                )

    def batch_alignment(self, from_class_iri: str, from_labels: List[str], search_space: List[str], flag: str):

        from_ob, to_ob = self.from_to_config(flag=flag)
        to_batch_size = max(self.batch_size // len(from_labels), self.nbest + 1)
        to_texts_iterator = to_ob.onto_text.batch_iterator(search_space, to_batch_size)
        j = 0
        batch_nbest_scores = torch.tensor([-1] * self.nbest).to(self.device)
        batch_nbest_idxs = torch.tensor([-1] * self.nbest).to(self.device)
        from_text_dict = from_ob.onto_text.texts[from_class_iri]
        from_embed = self.bert.ontotext_embeds({from_class_iri: from_text_dict}, strategy=self.strategy)

        for to_batch in to_texts_iterator:
            batch_label_pairs = []
            batch_lens = []
            # prepare a batch of label pairs for a given from-onto class
            for to_class_iri, text_dict in to_batch.items():
                to_labels = text_dict["label"]
                label_pairs = [[from_label, to_label] for to_label in to_labels for from_label in from_labels]
                # return the map if the to-class has a label that is exactly the same as one of the labels of the from-class
                if self.string_match:
                    for pair in label_pairs:
                        if pair[0] == pair[1]:
                            return [(to_class_iri, 1.0)]
                batch_label_pairs += label_pairs
                batch_lens.append(len(to_labels) * len(from_labels))

            # retrieve the batch embeds
            to_batch_embeds = self.bert.ontotext_embeds(to_batch, strategy=self.strategy)

            # compare the cosine similarity scores between two batches
            sim_scores = torch.tensor(cosine_similarity(from_embed, to_batch_embeds)).to(self.device).squeeze(0)
            K = len(sim_scores) if len(sim_scores) < self.nbest else self.nbest
            nbest_scores, nbest_idxs = torch.topk(sim_scores, k=K)
            nbest_idxs += j * len(to_batch)
            # we do the substituion for every batch to prevent from memory overflow
            batch_nbest_scores, temp_idxs = torch.topk(torch.cat([batch_nbest_scores, nbest_scores]), k=self.nbest)
            batch_nbest_idxs = torch.cat([batch_nbest_idxs, nbest_idxs])[temp_idxs]
            j += 1

        batch_nbest_class_iris = [search_space[idx] for idx in batch_nbest_idxs]
        return list(zip(batch_nbest_class_iris, batch_nbest_scores.cpu().detach().numpy()))
