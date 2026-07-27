# -*- coding: utf-8 -*-
"""Export a PROC_PARAMS sweep to a text-free CSV that is safe to commit.

cgp_out/ is gitignored because it contains model completions of adversarial prompts.
The numbers behind docs/proc_params_sweep.md are not sensitive, so this drops every
text field and keeps the per-row metrics, giving the write-up an in-repo data table.

usage: python sweep_export.py [proc_sweep.json] [out.csv]
"""
import csv, json, os, sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJ, "cgp_out", "proc_sweep_gemma-3-4b.json")
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PROJ, "docs", "proc_params_sweep_rows.csv")

d = json.load(open(src, encoding="utf-8"))
COLS = ["combo", "initial_bias", "max_bias", "ramp_steps", "grace_period",
        "idx", "dataset", "ptype", "base_avg", "bias_avg", "H", "p_ratio_norm",
        "identical", "base_refusal", "bias_refusal", "degenerate",
        "proc_calls", "proc_applied", "proc_max_bias", "proc_stop_step"]

os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(COLS)
    for r in d["results"]:
        p = r["params"]
        for row in r["rows"]:
            w.writerow([r["combo"], p["initial_bias"], p["max_bias"], p["ramp_steps"],
                        p["grace_period"], row["idx"], row["dataset"], row["ptype"],
                        round(row["base_avg"], 6), round(row["bias_avg"], 6),
                        round(row["H"], 6), round(row["p_ratio_norm"], 6),
                        int(row["identical"]), int(row["base_ref"]), int(row["bias_ref"]),
                        int(row["degen"]), row["proc_calls"], row["proc_applied"],
                        row["proc_max_bias"], row["proc_stop_step"]])

n = sum(len(r["rows"]) for r in d["results"])
print(f"wrote {dst}  ({n} rows x {len(COLS)} cols, {d['model']} {d['quant']}, no text fields)")
