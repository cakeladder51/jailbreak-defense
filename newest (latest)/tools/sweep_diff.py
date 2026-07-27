# -*- coding: utf-8 -*-
"""Row-by-row diff of a PROC_PARAMS sweep -- which rows actually responded?

The means printed by sweep_proc.py are the wrong unit of analysis. On gemma the mean
attack ratio appears to climb as the ramp is front-loaded, but the row-level view shows
only the two cipherchat_cipher rows moving at all below max_bias=10, non-monotonically,
with gibberish on both sides. Always look here before believing a combo is better.

usage: python sweep_diff.py [cgp_out/proc_sweep_<model>.json]
"""
import json, os, sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJ, "cgp_out", "proc_sweep_gemma-3-4b.json")

d = json.load(open(path, encoding="utf-8"))
res = d["results"]
names = [r["combo"] for r in res]
by = {r["combo"]: {row["idx"]: row for row in r["rows"]} for r in res}
idxs = sorted(by[names[0]])

print(f"{d['model']} ({d['quant']})  combos: {names}\n")
print(f"{'idx':<5}{'dataset':<20}{'ptype':<14}{'#texts':>7}   p_ratio_norm per combo")
n_moved = 0
for i in idxs:
    texts = [by[c][i]["bias_text"] for c in names]
    k = len(set(texts))
    n_moved += (k > 1)
    r0 = by[names[0]][i]
    ratios = " ".join(f"{by[c][i]['p_ratio_norm']:.3f}" for c in names)
    print(f"{i:<5}{r0['dataset'][:19]:<20}{r0['ptype']:<14}{k:>7}   {ratios}")

print(f"\n{n_moved}/{len(idxs)} rows changed text somewhere in the grid; "
      f"{len(idxs)-n_moved}/{len(idxs)} were unaffected by PROC_PARAMS entirely.")

# The question that actually matters: does a combo move ATTACK rows without moving BENIGN ones?
base = names[0]
print(f"\nrows whose text differs from '{base}':")
for c in names[1:]:
    atk = sum(1 for i in idxs
              if by[c][i]["ptype"] != "benign" and by[c][i]["bias_text"] != by[base][i]["bias_text"])
    ben = sum(1 for i in idxs
              if by[c][i]["ptype"] == "benign" and by[c][i]["bias_text"] != by[base][i]["bias_text"])
    n_a = sum(1 for i in idxs if by[base][i]["ptype"] != "benign")
    n_b = sum(1 for i in idxs if by[base][i]["ptype"] == "benign")
    print(f"  {c:<11} attack {atk:>2}/{n_a:<3} benign {ben:>2}/{n_b:<3}"
          f"   <- want high attack, low benign")
