# -*- coding: utf-8 -*-
"""Per-family AUC for the headline signal AND its controls, side by side."""
import json, sys, os
import numpy as np

path = sys.argv[1]
rows = []
for line in open(path, encoding="utf-8"):
    try:
        r = json.loads(line)
        if "error" not in r: rows.append(r)
    except Exception: pass

ATK = ("harmful", "forced_prefix")
def _auc(pos, neg):
    pos = np.asarray([v for v in pos if v is not None], float)
    neg = np.asarray([v for v in neg if v is not None], float)
    if not len(pos) or not len(neg): return None
    allv = np.concatenate([pos, neg]); order = allv.argsort(kind="mergesort"); sv = allv[order]
    ranks = np.empty(len(allv), float); ranks[order] = np.arange(1, len(allv)+1)
    i = 0
    while i < len(sv):
        j = i
        while j+1 < len(sv) and sv[j+1] == sv[i]: j += 1
        if j > i: ranks[order[i:j+1]] = (i+1 + j+1)/2.0
        i = j+1
    return float((ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2.0) / (len(pos)*len(neg)))

ben = [r for r in rows if r["prompt_type"] == "benign"]
atk = [r for r in rows if r["prompt_type"] in ATK]
KEYS = ["p_ratio_norm", "base_avg", "H"]

print(os.path.basename(path))
print("\nAUC per family. Values are |AUC-0.5| oriented: 0.5 = chance, distance from 0.5 = signal.")
print(f"\n{'family':26s}" + "".join(f"{k:>16s}" for k in KEYS))
print("-" * (26 + 16*len(KEYS)))
for ds in sorted({r.get("dataset") for r in atk}):
    sub = [r for r in atk if r.get("dataset") == ds]
    line = f"{ds:26s}"
    for k in KEYS:
        a = _auc([r.get(k) for r in sub], [r.get(k) for r in ben])
        line += f"{a:>10.3f} ({abs(2*a-1):.2f})" if a is not None else f"{'-':>16s}"
    print(line)
line = f"{'ALL ATTACKS':26s}"
for k in KEYS:
    a = _auc([r.get(k) for r in atk], [r.get(k) for r in ben])
    line += f"{a:>10.3f} ({abs(2*a-1):.2f})" if a is not None else f"{'-':>16s}"
print("-" * (26 + 16*len(KEYS))); print(line)

print("\nClass means (raw scale):")
print(f"{'group':26s}{'base_avg':>12s}{'bias_avg':>12s}{'p_ratio_norm':>14s}{'H':>10s}")
def m(rs, k):
    v = [r.get(k) for r in rs if r.get(k) is not None]
    return np.mean(v) if v else float("nan")
for label, rs in [("benign", ben), ("all attacks", atk)] + \
                 [(ds, [r for r in atk if r.get("dataset") == ds])
                  for ds in sorted({r.get("dataset") for r in atk})]:
    print(f"{label:26s}{m(rs,'base_avg'):12.3f}{m(rs,'bias_avg'):12.3f}"
          f"{m(rs,'p_ratio_norm'):14.3f}{m(rs,'H'):10.3f}")
