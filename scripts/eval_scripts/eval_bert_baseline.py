import sys
sys.path.append("/home/yuahe/projects/BERTMap")
from bertmap.map import OntoMapping
import pandas as pd
import multiprocessing

ref_dir = "/home/yuahe/projects/BERTMap/data/largebio/refs"
map_dir = "/home/yuahe/projects/BERTMap/experiment/bert_baseline/maps"

for name in ["mean", "cls"]:
    for src, tgt in [("fma", "nci"), ("fma", "snomed"), ("snomed", "nci")]:
        report = pd.DataFrame(columns=["Precision", "Recall", "F1", "#Illegal"])
        ref_legal = f"{ref_dir}/{src}2{tgt}.legal.tsv"
        ref_illegal = f"{ref_dir}/{src}2{tgt}.illegal.tsv"
        pool = multiprocessing.Pool(14) 
        eval_results = []
        #  0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
            eval_results.append(pool.apply_async(OntoMapping.evaluate, args=(f"{map_dir}/combined.{src}2{tgt}.small.bc-{name}.tsv", 
                                                                                     ref_legal, ref_illegal, f"combined", threshold)))
            eval_results.append(pool.apply_async(OntoMapping.evaluate, args=(f"{map_dir}/src2tgt.{src}2{tgt}.small.bc-{name}.tsv", 
                                                                                     ref_legal, ref_illegal, f"{src}2{tgt}", threshold)))
            eval_results.append(pool.apply_async(OntoMapping.evaluate, args=(f"{map_dir}/tgt2src{src}2{tgt}.small.bc-{name}.tsv", 
                                                                                     ref_legal, ref_illegal, f"{tgt}2{src}", threshold)))
        pool.close()
        pool.join()
        
        for result in eval_results:
            result = result.get()
            report = report.append(result)

        print(report)
        report.to_csv(f"{map_dir}/../results/bc-{name}.eval.{src}2{tgt}.small.csv")