import sys
sys.path.append("/home/yuahe/projects/BERTMap")
from bertmap.map import OntoMapping
import pandas as pd
import multiprocessing

for src, tgt in [("fma", "nci"), ("fma", "snomed"), ("snomed", "nci")]:
    report = pd.DataFrame(columns=["Precision", "Recall", "F1", "#Illegal"])
    dir_name = f"/home/yuahe/projects/BERTMap/largebio_exp/small/{src}2{tgt}"
    ref_legal = f"/home/yuahe/projects/BERTMap/largebio_data/references/{src}2{tgt}.legal.tsv"
    ref_illegal = f"/home/yuahe/projects/BERTMap/largebio_data/references/{src}2{tgt}.illegal.tsv"
    pool = multiprocessing.Pool(14) 
    eval_results = []
    for threshold in [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        eval_results.append(pool.apply_async(OntoMapping.evaluate, args=(f"{dir_name}/{src}-{tgt}-combined.small.nes.tsv", 
                                                                                    ref_legal, ref_illegal, f"combined", threshold)))
        eval_results.append(pool.apply_async(OntoMapping.evaluate, args=(f"{dir_name}/{src}2{tgt}.small.nes.tsv", 
                                                                                    ref_legal, ref_illegal, f"{src}2{tgt}", threshold)))
        eval_results.append(pool.apply_async(OntoMapping.evaluate, args=(f"{dir_name}/{tgt}2{src}.small.nes.tsv", 
                                                                                    ref_legal, ref_illegal, f"{tgt}2{src}", threshold)))
    pool.close()
    pool.join()
    
    for result in eval_results:
        result = result.get()
        report = report.append(result)

    print(report)
    report.to_csv(f"{dir_name}/nes.eval.csv")