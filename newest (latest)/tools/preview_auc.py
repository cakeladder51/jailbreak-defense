# -*- coding: utf-8 -*-
"""Preview A1 separation straight from a banked *_cgp.jsonl (no judge, no GPU).

Reports **oriented** AUC alongside the raw value. Raw AUC is computed with attacks as the
positive class, so a value BELOW 0.5 means attacks score lower -- which for p_ratio_norm is
the direction the method predicts ("gap below threshold -> serve the refusal") and which
fit_threshold picks up automatically as direction=-1. Reading the raw number as if higher
were always better turns a working signal into an apparent failure, so both are shown.

When the run has more than one benign source it also breaks separation out per source:
alpaca is the EASY benign set, xstest_safe the HARD one (prompts that look harmful but are
not). The gap between those two columns is what distinguishes a safety signal from an
"unusual text" detector.
"""
import json, sys, glob, os
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = sys.argv[1] if len(sys.argv) > 1 else None
if not path:
    path = max(glob.glob(os.path.join(PROJ, "cgp_out", "*_cgp.jsonl")), key=os.path.getmtime)
rows = []
for line in open(path, encoding="utf-8"):
    try:
        r = json.loads(line)
        if "error" not in r: rows.append(r)
    except Exception: pass

ATK = ("harmful", "forced_prefix")
SIGNALS = [("p_ratio_norm",    "headline"),
           ("refusal_then_gap", "candidate"),
           ("bias_is_refusal",  "candidate"),
           ("p_ratio_signed",  ""),
           ("p_ratio_raw",     ""),
           ("bias_avg",        ""),
           ("base_avg",        "CONTROL"),
           ("H",               "CONTROL")]


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


def orient(a):
    return None if a is None else max(a, 1.0 - a)


def ci(pos, neg, n=2000, seed=0):
    pos = [v for v in pos if v is not None]; neg = [v for v in neg if v is not None]
    a = _auc(pos, neg)
    if a is None: return (None, None, None)
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        v = _auc([pos[i] for i in rng.integers(0, len(pos), len(pos))],
                 [neg[i] for i in rng.integers(0, len(neg), len(neg))])
        if v is not None: vals.append(orient(v))
    return orient(a), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


ben = [r for r in rows if r["prompt_type"] == "benign"]
atk = [r for r in rows if r["prompt_type"] in ATK]
bsrc = sorted({r.get("dataset") for r in ben if r.get("dataset")})
mix = ", ".join(f"{s} {sum(1 for r in ben if r.get('dataset') == s)}" for s in bsrc)
print(f"{os.path.basename(path)}: {len(rows)} rows")
print(f"n_attack={len(atk)}  n_benign={len(ben)}  ({mix})\n")

print(f"{'signal':17s}{'raw':>7}{'oriented':>10}{'95% CI':>17}  {'direction':<15} note")
print("-" * 78)
avail = [(k, tag) for k, tag in SIGNALS if any(r.get(k) is not None for r in rows)]
for k, tag in avail:
    raw = _auc([r.get(k) for r in atk], [r.get(k) for r in ben])
    if raw is None: continue
    o, lo, hi = ci([r.get(k) for r in atk], [r.get(k) for r in ben])
    d = "attacks-lower" if raw < 0.5 else "attacks-higher"
    print(f"{k:17s}{raw:>7.3f}{o:>10.3f}{f'[{lo:.3f},{hi:.3f}]':>17}  {d:<15} {tag}")

if len(bsrc) > 1:
    print(f"\n{'=' * 78}\nEASY vs HARD BENIGN -- oriented AUC of attacks against each benign set")
    print("a signal that only detects unusual text collapses in the hard column")
    print(f"{'=' * 78}")
    hdr = "".join(f"{s[:14]:>16}" for s in bsrc)
    print(f"{'signal':17s}{hdr}{'drop':>9}")
    print("-" * 78)
    for k, tag in avail:
        vals = []
        for s in bsrc:
            sub = [r for r in ben if r.get("dataset") == s]
            vals.append(orient(_auc([r.get(k) for r in atk], [r.get(k) for r in sub])))
        if any(v is None for v in vals): continue
        easy = vals[bsrc.index("alpaca")] if "alpaca" in bsrc else max(vals)
        hard = vals[bsrc.index("xstest_safe")] if "xstest_safe" in bsrc else min(vals)
        cells = "".join(f"{v:>16.3f}" for v in vals)
        print(f"{k:17s}{cells}{hard - easy:>+9.3f}  {tag}")

best = avail[0][0]
print(f"\nper-family oriented AUC ({best} vs all benign):")
for ds in sorted({r.get("dataset") for r in atk}):
    sub = [r for r in atk if r.get("dataset") == ds]
    a = orient(_auc([r.get(best) for r in sub], [r.get(best) for r in ben]))
    print(f"  {ds:26s} {a:6.3f}  (n={len(sub)})" if a is not None else f"  {ds:26s}   -")

print("\nprocessor / cost:")
for k in ("proc_applied", "proc_stop_step", "t_base", "t_bias"):
    v = [r.get(k) for r in rows if r.get(k) is not None]
    if v: print(f"  {k:16s} mean={np.mean(v):8.2f}  median={np.median(v):8.2f}")
fired = [r for r in rows if (r.get("proc_applied") or 0) > 0]
print(f"  processor fired on {len(fired)}/{len(rows)} rows")
ident = sum(1 for r in rows if r.get("base_text") == r.get("bias_text"))
print(f"  biased text identical to baseline on {ident}/{len(rows)} rows "
      f"({100.0*ident/max(len(rows),1):.0f}%)")
