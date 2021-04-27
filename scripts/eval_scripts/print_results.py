import pandas as pd
import sys

eval_csv = sys.argv[1]
df = pd.read_csv(eval_csv, index_col=0)
print("------------ Evaluation Results ------------")
print(df)
print("------------ Best String Match Results ------------")
print(df.loc[df["F1"][-3:].idxmax()])
print("------------ Best Results without String Match ------------")
print(df.loc[df["F1"][:-3].idxmax()])