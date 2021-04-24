import os
main_dir = os.getcwd().split("BERTMap")[0] + "BERTMap"
import sys
sys.path.append(main_dir)
from bertmap.map import OntoMapping
from bertmap.utils import evenly_divide
import pandas as pd
import multiprocessing

src = sys.argv[1]
tgt = sys.argv[2]
task = sys.argv[3]
num_pool = int(sys.argv[4])
ref_dir = f"{main_dir}/data/largebio/refs"
report = pd.DataFrame(columns=["Precision", "Recall", "F1", "#Illegal"])
ref_legal = f"{ref_dir}/{src}2{tgt}.legal.tsv"
ref_illegal = f"{ref_dir}/{src}2{tgt}.illegal.tsv"
pool = multiprocessing.Pool(num_pool) 
eval_results = []
map_dir = f"{main_dir}/experiment/bert_baseline/{src}2{tgt}/{task}"
for threshold in [0.0, 0.3, 0.5, 0.7, 0.9, 0.92] + evenly_divide(0.95, 1.0, int(sys.argv[5])):
    threshold = round(threshold, 6)
    eval_results.append(pool.apply_async(OntoMapping.evaluate, args=(f"{map_dir}/combined.maps.tsv", ref_legal, ref_illegal, f"combined", threshold)))
    eval_results.append(pool.apply_async(OntoMapping.evaluate, args=(f"{map_dir}/src2tgt.maps.tsv", ref_legal, ref_illegal, f"{src}2{tgt}", threshold)))
    eval_results.append(pool.apply_async(OntoMapping.evaluate, args=(f"{map_dir}/tgt2src.maps.tsv", ref_legal, ref_illegal, f"{tgt}2{src}", threshold)))
pool.close()
pool.join()

for result in eval_results:
    result = result.get()
    report = report.append(result)

print(report)
max_scores = list(report.max()[["Precision", "Recall", "F1"]])
max_inds = list(report.idxmax()[["Precision", "Recall", "F1"]])
min_illegal = list(report.min()[["#Illegal"]])
print(f"Best results are: P: {max_scores[0]} ({max_inds[0]}); R: {max_scores[1]} ({max_inds[1]}); \
    F1: {max_scores[2]} ({max_inds[2]}); #Illegal: {min_illegal[0]}.")

report.to_csv(f"{map_dir}/eval.csv")