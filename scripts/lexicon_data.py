import sys
sys.path.append("/home/yuahe/projects/BERTMap")
from bertmap.onto import Ontology

data_base = "/home/yuahe/projects/BERTMap/largebio_data/ontologies"
save_path= f"{data_base}/../onto_labels"
properties = ["label"]

for src, tgt in [("fma", "nci"), ("fma", "snomed"), ("snomed", "nci")]:
    src_onto = Ontology(f"{data_base}/{src}2{tgt}.small.owl")
    tgt_onto = Ontology(f"{data_base}/{tgt}2{src}.small.owl")
    src_onto.class2text_df(*properties).to_csv(f"{save_path}/{src}2{tgt}.small.labels.tsv", sep="\t", index=False)
    tgt_onto.class2text_df(*properties).to_csv(f"{save_path}/{tgt}2{src}.small.labels.tsv", sep="\t", index=False)
    
for name in ["fma", "nci", "snomed"]:
    onto = Ontology(f"{data_base}/{name}.whole.owl")
    # onto = Ontology(f"{data_base}/../largebio_raw/oaei_FMA_whole_ontology.owl")
    # print(len(list(onto.onto.classes())))
    onto.class2text_df(*properties).to_csv(f"{save_path}/{name}.whole.labels.tsv", sep="\t", index=False)
