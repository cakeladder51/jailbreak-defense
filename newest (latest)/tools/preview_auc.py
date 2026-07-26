# -*- coding: utf-8 -*-
"""Preview A1 separation straight from a banked *_cgp.jsonl (no judge, no GPU)."""
import json, sys, glob, os
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else None
if not path:
    cands = glob.glob(r"D:\CodeFiles\AI project\newest (latest)\cgp_out\*_cgp.jsonl")
    path = max(cands, key=os.path.getmtime)
rows = []
for line in open(path, encoding="utf-8"):
    try:
        r = json.loads(line)
        if "error" not in r: rows.append(r)
    except Exception: pass
print(f"{os.path.basename(path)}: {len(rows)} rows\n")

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

def ci(pos, neg, n=2000, seed=0):
    pos = [v for v in pos if v is not None]; neg = [v for v in neg if v is not None]
    a = _auc(pos, neg)
    if a is None: return (None, None, None)
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        v = _auc([pos[i] for i in rng.integers(0, len(pos), len(pos))],
                 [neg[i] for i in rng.integers(0, len(neg), len(neg))])
        if v is not None: vals.append(v)
    return a, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

ben = [r for r in rows if r["prompt_type"] == "benign"]
atk = [r for r in rows if r["prompt_type"] in ATK]
print(f"n_attack={len(atk)}  n_benign={len(ben)}\n")
print(f"{'signal':16s} {'AUC':>6s} {'95% CI':>16s} {'|delta|':>8s}   note")
print("-" * 68)
for k in ("p_ratio_norm", "p_ratio_signed", "p_ratio_raw", "base_avg", "bias_avg", "H", "H_bias"):
    if not any(r.get(k) is not None for r in rows): continue
    a, lo, hi = ci([r.get(k) for r in atk], [r.get(k) for r in ben])
    if a is None: continue
    note = "<== HEADLINE" if k == "p_ratio_norm" else ("(control)" if k in ("base_avg", "H") else "")
    print(f"{k:16s} {a:6.3f} {f'[{lo:.3f},{hi:.3f}]':>16s} {abs(2*a-1):8.3f}   {note}")

head = _auc([r.get("p_ratio_norm") for r in atk], [r.get("p_ratio_norm") for r in ben])
print("\nper-family AUC (p_ratio_norm, family vs all benign):")
for ds in sorted({r.get("dataset") for r in atk}):
    sub = [r for r in atk if r.get("dataset") == ds]
    a = _auc([r.get("p_ratio_norm") for r in sub], [r.get("p_ratio_norm") for r in ben])
    print(f"  {ds:26s} {a:6.3f}  (n={len(sub)})" if a is not None else f"  {ds:26s}   -")

print("\nprocessor / cost:")
for k in ("proc_applied", "proc_stop_step", "t_base", "t_bias"):
    v = [r.get(k) for r in rows if r.get(k) is not None]
    if v: print(f"  {k:16s} mean={np.mean(v):8.2f}  median={np.median(v):8.2f}")
fired = [r for r in rows if (r.get("proc_applied") or 0) > 0]
print(f"  processor fired on {len(fired)}/{len(rows)} rows")
