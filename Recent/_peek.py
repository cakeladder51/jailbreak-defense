import pandas as pd
df = pd.read_csv("results_parallel/results_Qwen2.5-3B.csv")
print("rows:", len(df))
print("cols ok:", "p_ratio_norm" in df.columns)
print(df[["prompt_type", "baseline_prob", "logit_biasing_prob", "p_ratio_norm"]].head(6).to_string())
print("by type so far:")
print(df.groupby("prompt_type")["p_ratio_norm"].agg(["count", "mean"]))
