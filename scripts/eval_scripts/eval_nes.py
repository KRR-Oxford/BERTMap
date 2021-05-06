"""
Will not run this experiment again due to its time complexity
"""

import sys
sys.path.append("/home/yuahe/projects/BERTMap")
from bertmap.map import OntoMapping
import pandas as pd
import multiprocessing

ref_dir = "/home/yuahe/projects/BERTMap/data/largebio/refs"
map_dir = "/home/yuahe/projects/BERTMap/experiment/nes_baseline/maps"

for src, tgt in [("fma", "nci"), ("fma", "snomed"), ("snomed", "nci")]:
    report = pd.DataFrame(columns=["Precision", "Recall", "F1", "#Illegal"])
    ref_legal = f"{ref_dir}/{src}2{tgt}.legal.tsv"
    ref_illegal = f"{ref_dir}/{src}2{tgt}.illegal.tsv"
    pool = multiprocessing.Pool(14) 
    eval_results = []
    for threshold in [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        eval_results.append(pool.apply_async(OntoMapping.evaluate, args=(f"{map_dir}/combined.{src}2{tgt}.small.nes.tsv", 
                                                                                    ref_legal, ref_illegal, f"combined", threshold)))
        eval_results.append(pool.apply_async(OntoMapping.evaluate, args=(f"{map_dir}/src2tgt.{src}2{tgt}.small.nes.tsv", 
                                                                                    ref_legal, ref_illegal, f"{src}2{tgt}", threshold)))
        eval_results.append(pool.apply_async(OntoMapping.evaluate, args=(f"{map_dir}/tgt2src.{src}2{tgt}.small.nes.tsv", 
                                                                                    ref_legal, ref_illegal, f"{tgt}2{src}", threshold)))
    pool.close()
    pool.join()
    
    for result in eval_results:
        result = result.get()
        report = report.append(result)

    print(report)
    report.to_csv(f"{map_dir}/../results/nes.eval.{src}2{tgt}.csv")