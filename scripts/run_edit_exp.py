import sys
sys.path.append("/home/yuahe/projects/OntoAlign-py")
from ontoalign.experiments.direct_search import DirectNormEditSimExperiment
import multiprocessing

for src, tgt in [("fma", "nci"), ("fma", "snomed"), ("snomed", "nci")]:
    
    base = "/home/yuahe/projects/OntoAlign-py/largebio_data/onto_labels"
    src_onto_lexicon_tsv = f"{base}/{src}2{tgt}.small.labels.tsv"
    tgt_onto_lexicon_tsv = f"{base}/{tgt}2{src}.small.labels.tsv"
    
    exp = DirectNormEditSimExperiment(
                            src, tgt, 
                            src_onto_lexicon_tsv, tgt_onto_lexicon_tsv,
                            f"/home/yuahe/projects/OntoAlign-py/largebio_exp/small/{src}2{tgt}/",
                            task_suffix="small", num_pools=20)
    exp.run()
    exp.save()

