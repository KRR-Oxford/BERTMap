main_dir = "/home/lawhy0729/BERTMap"
import sys
sys.path.append(main_dir)
import pandas as pd
from bertmap.corpora import CrossOntoCorpus

src = "fma"
tgt = "nci"
src_label_path = main_dir + f"/data/largebio/labels/{src}2{tgt}.small.labels.tsv"
tgt_label_path = main_dir + f"/data/largebio/labels/{tgt}2{src}.small.labels.tsv"
src_onto_path = main_dir + f"/data/largebio/ontos/{src}2{tgt}.small.owl"
tgt_onto_path = main_dir + f"/data/largebio/ontos/{tgt}2{src}.small.owl"
known_map_path = main_dir + f"/data/largebio/refs/{src}2{tgt}.legal.tsv"
co = CrossOntoCorpus("fma2nci", src_onto_path, tgt_onto_path, known_map_path, src_label_path, tgt_label_path)
co.create_class2text_dict(co.src_onto_class2text)
fixed_src_class2text = pd.DataFrame(columns=["Class-IRI", "Class-Text"])
fixed_src_classes = []
fixed_src_texts = []
for i in range(len(co.maps)):
    ref_map = co.maps[i]
    src_class, tgt_class = ref_map.split("\t")
    fixed_src_classes.append(src_class)
    fixed_src_texts.append(co.src_onto_class2text_dict[src_class])
fixed_src_class2text["Class-IRI"] = fixed_src_classes
fixed_src_class2text["Class-Text"] = fixed_src_texts
fixed_src_class2text.to_csv(f"{main_dir}/{src}_labels_from_maps.tsv", sep="\t", index=False)