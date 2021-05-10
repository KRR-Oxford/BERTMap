"""
Corpora class for handling all kinds of sub-corpora involved in an alignment task
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from bertmap.corpora import CrossOntoCorpus, IntraOntoCorpus, MergedOntoCorpus
from bertmap.onto import OntoBox, OntoEvaluator
from bertmap.utils import uniqify


class OntoAlignCorpora:
    def __init__(
        self,
        src_ob: Optional[OntoBox] = None,
        tgt_ob: Optional[OntoBox] = None,
        src2tgt_mappings_file: Optional[Union[str, List[str]]] = None,
        ignored_mappings_file: Optional[str] = None,
        train_map_ratio: float = 0.2,
        val_map_ratio: float = 0.1,
        test_map_ratio: float = 0.7,
        sample_rate: int = 10,
        io_soft_neg_rate: int = 1,
        io_hard_neg_rate: int = 1,
        co_soft_neg_rate: int = 2,
        depth_threshold: Optional[int] = None,
        depth_strategy: Optional[str] = "max",
        from_saved: bool = False,
    ):
        if not from_saved:
            # reference mappings
            self.maps = OntoEvaluator.read_mappings(src2tgt_mappings_file)
            if ignored_mappings_file:
                self.ignored_maps = OntoEvaluator.read_mappings(ignored_mappings_file)
            else:
                self.ignored_maps = []
            # attributes for extracting label data
            self.tra_map_ratio = train_map_ratio
            self.val_map_ratio = val_map_ratio
            self.test_map_ratio = test_map_ratio
            self.io_soft_neg_rate = io_soft_neg_rate
            self.io_hard_neg_rate = io_hard_neg_rate
            self.co_soft_neg_rate = co_soft_neg_rate
            # intra-onto copora merged
            self.src_io = IntraOntoCorpus(src_ob, sample_rate, depth_threshold, depth_strategy)
            self.tgt_io = IntraOntoCorpus(tgt_ob, sample_rate, depth_threshold, depth_strategy)
            self.src_tgt_io = MergedOntoCorpus(self.src_io, self.tgt_io)
            # semi-supervised cross-onto corpus
            self.train_ss_co, self.val_ss_co, self.test_ss_co = CrossOntoCorpus.spliting_corpus(
                src_ob, tgt_ob, src2tgt_mappings_file, train_map_ratio, val_map_ratio, test_map_ratio, sample_rate
            )
            # unsupervised cross-onto corpus
            self.src_tgt_co = CrossOntoCorpus(src_ob, tgt_ob, src2tgt_mappings_file, sample_rate)
        else:
            pass

    def __repr__(self):
        report = "<OntoAlignCorpora>\n"
        report += f"\t<RefMaps full={len(self.maps)} ignored={len(self.ignored_maps)}>\n"
        report += (
            "\t<!-- NegRate refers to the number of (soft/hard) nonsynonym sampled for each synonym in a corpus -->\n"
        )
        report += f"\t<NegRate intra_onto_soft={self.io_soft_neg_rate} intra_onto_hard={self.io_hard_neg_rate} cross_onto_soft={self.co_soft_neg_rate}>\n"
        report += "\n\t<!-- merged intra-onto corpus -->"
        report += f"\n{str(self.src_tgt_io)}".replace("\n", "\n\t")
        report += f"\n\n\t<!-- if semi-supervised, the reference mappings will be split as follows: -->"
        report += f"\n\t<SemiSupervisedSetting train={self.tra_map_ratio} val={self.val_map_ratio} test={self.test_map_ratio}>"
        report += "\n\t\t<!-- train cross-onto corpus (if semi-supervised) -->"
        report += f"\n{str(self.train_ss_co)}".replace("\n", "\n\t\t")
        report += "\n\t\t<!-- val cross-onto corpus (if semi-supervised) -->"
        report += f"\n{str(self.val_ss_co)}".replace("\n", "\n\t\t")
        report += "\n\t\t<!-- test cross-onto corpus (if semi-supervised) -->"
        report += f"\n{str(self.test_ss_co)}".replace("\n", "\n\t\t")
        report += f"\n\t</SemiSupervisedSetting>"
        report += f"\n\n\t<UnSupervisedSetting test=1.0>"
        report += "\n\t\t<!-- test cross-onto corpus (if un-supervised) -->"
        report += f"\n{str(self.src_tgt_co)}".replace("\n", "\n\t\t")
        report += f"\n\t</UnSupervisedSetting>\n"
        report += "</OntoAlignCorpora>"
        return report

    def save(self, save_dir) -> None:
        Path(save_dir + "/refs").mkdir(parents=True, exist_ok=True)
        Path(save_dir + "/corpora").mkdir(parents=True, exist_ok=True)
        # copy and save the reference mappings for record
        self.save_maps(self.maps, save_dir + "/refs/maps.ref.us.tsv")
        self.save_maps(self.train_ss_co.maps, save_dir + "/refs/maps.ref.ss.train.tsv")
        self.save_maps(self.val_ss_co.maps, save_dir + "/refs/maps.ref.ss.val.tsv")
        self.save_maps(self.test_ss_co.maps, save_dir + "/refs/maps.ref.ss.test.tsv")
        self.save_maps(self.ignored_maps, save_dir + "/refs/maps.ignored.tsv")
        # save the corpora
        self.src_tgt_io.save_corpus(save_dir + "/corpora/io.corpus.json")
        self.train_ss_co.save_corpus(save_dir + "/corpora/co.corpus.ss.train.json")
        self.val_ss_co.save_corpus(save_dir + "/corpora/co.corpus.ss.val.json")
        self.test_ss_co.save_corpus(save_dir + "/corpora/co.corpus.ss.test.json")
        self.src_tgt_co.save_corpus(save_dir + "/corpora/co.corpus.us.json")
        # save the corpora info
        with open(save_dir + "/corpora/info", "w") as f:
            f.write(str(self))

    def save_maps(self, loaded_maps, save_file) -> None:
        maps = [(map.split("\t")[0], map.split("\t")[1], 1.0) for map in loaded_maps]
        pd.DataFrame(maps, columns=["Entity1", "Entity2", "Value"]).to_csv(save_file, sep="\t", index=False)

    @classmethod
    def from_saved(cls, save_dir) -> OntoAlignCorpora:
        oa_corpora = cls(from_saved=True)
        oa_corpora.src_tgt_io = MergedOntoCorpus(corpus_file=save_dir + "/corpora/io.corpus.json")
        oa_corpora.train_ss_co = CrossOntoCorpus(corpus_file=save_dir + "/corpora/co.corpus.ss.train.json")
        oa_corpora.val_ss_co = CrossOntoCorpus(corpus_file=save_dir + "/corpora/co.corpus.ss.val.json")
        oa_corpora.test_ss_co = CrossOntoCorpus(corpus_file=save_dir + "/corpora/co.corpus.ss.test.json")
        oa_corpora.src_tgt_co = CrossOntoCorpus(corpus_file=save_dir + "/corpora/co.corpus.us.json")
        with open(save_dir + "/corpora/info", "r") as f:
            lines = f.readlines()
        neg_rates = re.findall(
            r"<NegRate intra_onto_soft=([0-9]+) intra_onto_hard=([0-9]+) cross_onto_soft=([0-9]+)>", lines[3]
        )[0]
        oa_corpora.io_soft_neg_rate = int(neg_rates[0])
        oa_corpora.io_hard_neg_rate = int(neg_rates[1])
        oa_corpora.co_soft_neg_rate = int(neg_rates[2])
        for line in lines:
            if re.findall(r"<SemiSupervisedSetting train=(.+) val=(.+) test=(.+)>", line):
                map_ratios = re.findall(r"<SemiSupervisedSetting train=(.+) val=(.+) test=(.+)>", line)[0]
                oa_corpora.tra_map_ratio = float(map_ratios[0])
                oa_corpora.val_map_ratio = float(map_ratios[1])
                oa_corpora.test_map_ratio = float(map_ratios[2])
        oa_corpora.maps = OntoEvaluator.read_mappings(save_dir + "/refs/maps.ref.us.tsv")
        oa_corpora.ignored_maps = OntoEvaluator.read_mappings(save_dir + "/refs/maps.ignored.tsv")
        return oa_corpora

    def semi_supervised_data(
        self, io_soft_neg_rate: int, io_hard_neg_rate: int, co_soft_neg_rate: int, **kwargs
    ) -> Tuple[Dict[str, List[str]], str]:
        ss_io_train, ss_io_train_ids = self.src_tgt_io.train_val_split(
            val_ratio=0.0, soft_neg_rate=io_soft_neg_rate, hard_neg_rate=io_hard_neg_rate
        )
        # train-val-test data from mappings
        ss_co_train = self.train_ss_co.generate_label_data(soft_neg_rate=co_soft_neg_rate)
        ss_co_val = self.val_ss_co.generate_label_data(soft_neg_rate=co_soft_neg_rate)
        ss_co_test = self.test_ss_co.generate_label_data(soft_neg_rate=co_soft_neg_rate)
        report = f"data sizes before merging and duplicates removal ...\n"
        report += f"\tio_train: {len(ss_io_train)}\n"
        report += f"\tio_train_ids: {len(ss_io_train_ids)}\n"
        report += f"\tco_train: {len(ss_co_train)}, co_val: {len(ss_co_val)} co_test: {len(ss_co_test)}\n"

        ss_train = uniqify(ss_io_train + ss_co_train)
        ss_train_with_ids = uniqify(ss_io_train + ss_co_train + ss_io_train_ids)
        random.shuffle(ss_train)
        random.shuffle(ss_train_with_ids)
        random.shuffle(ss_co_val)
        random.shuffle(ss_co_test)
        ss_data = {"train": ss_train, "train+": ss_train_with_ids, "val": ss_co_val, "test": ss_co_test}
        report += "generate semi-supervised data files with following sizes ...\n"
        for k, v in ss_data.items():
            report += f"\t{k}: {len(v)}\n"
        print(report)
        return ss_data, report

    def unsupervised_data(
        self, io_soft_neg_rate: int, io_hard_neg_rate: int, co_soft_neg_rate: int, **kwargs
    ) -> Tuple[Dict[str, List[str]], str]:
        us_io_train, us_io_val, us_io_train_ids, us_io_val_ids = self.src_tgt_io.train_val_split(
            val_ratio=0.2, soft_neg_rate=io_soft_neg_rate, hard_neg_rate=io_hard_neg_rate
        )
        us_co_test = self.src_tgt_co.generate_label_data(soft_neg_rate=co_soft_neg_rate)
        report = f"data sizes before merging and duplicates removal ...\n"
        report += f"\tus_io_train: {len(us_io_train)}\n"
        report += f"\tus_io_train_ids: {len(us_io_train_ids)}\n"
        report += f"\tus_io_val: {len(us_io_val)}\n"
        report += f"\tus_io_val: {len(us_io_val)}\n"
        report += f"\tus_io_val_ids: {len(us_io_val_ids)}\n"

        us_train_with_ids = uniqify(us_io_train + us_io_train_ids)
        us_val_with_ids = uniqify(us_io_val + us_io_val_ids)
        random.shuffle(us_io_train)
        random.shuffle(us_train_with_ids)
        random.shuffle(us_io_val)
        random.shuffle(us_val_with_ids)
        random.shuffle(us_co_test)
        us_data = {
            "train": us_io_train,
            "train+": us_train_with_ids,
            "val": us_io_val,
            "val+": us_val_with_ids,
            "test": us_co_test,
        }
        report += "generate unsupervised data files with following sizes ...\n"
        for k, v in us_data.items():
            report += f"\t{k}: {len(v)}\n"
        print(report)
        return us_data, report
