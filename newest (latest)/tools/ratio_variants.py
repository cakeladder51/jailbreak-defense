# -*- coding: utf-8 -*-
"""Score alternative plausibility-gap definitions from banked per-token logprobs. No GPU.

`p_ratio_norm` is |mean logP(base_text) - mean logP(bias_text)| over the FULL 256-token
generations. Two things are suspect about that:
  * the refusal decision is made in the opening few tokens, and a 256-token mean drowns it
    in whatever the continuation happened to be;
  * a mean is dominated by a handful of very low-probability tokens, which is exactly what
    a degenerate biased generation produces (the cipherchat failure mode).
STORE_LOGPROBS=True banked `base_lps` / `bias_lps` per row, so every variant below can be
tested on the existing runs without regenerating anything.

Reported as ORIENTED AUC (max(a, 1-a)) against each benign source separately -- a variant
that only wins against easy benign is a text-weirdness detector, not an improvement.

usage: python ratio_variants.py [<cgp.jsonl> ...]     (default: all *_hb_cgp.jsonl)
"""
import glob, json, os, sys
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATK = ("harmful", "forced_prefix")

paths = sys.argv[1:] or sorted(glob.glob(os.path.join(PROJ, "cgp_out", "*_hb_cgp.jsonl")))
if not paths:
    sys.exit("no *_hb_cgp.jsonl found -- pass paths explicitly")


def auc(pos, neg):
    pos = np.asarray([v for v in pos if v is not None], float)
    neg = np.asarray([v for v in neg if v is not None], float)
    if not len(pos) or not len(neg): return None
    a = np.concatenate([pos, neg]); o = a.argsort(kind="mergesort"); sv = a[o]
    rk = np.empty(len(a), float); rk[o] = np.arange(1, len(a) + 1)
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]: j += 1
        if j > i: rk[o[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def orient(a):
    return None if a is None else max(a, 1.0 - a)


def _m(xs, k=None, how="mean"):
    """Summarise a logprob array: first k tokens, by mean / median / 10% trimmed mean."""
    if not xs: return None
    v = np.asarray(xs[:k] if k else xs, float)
    if not len(v): return None
    if how == "median": return float(np.median(v))
    if how == "trim":
        lo, hi = np.percentile(v, [10, 90])
        w = v[(v >= lo) & (v <= hi)]
        return float(w.mean()) if len(w) else float(v.mean())
    return float(v.mean())


def variants(r):
    """name -> scalar. All are |base - bias| unless the name says signed."""
    b, x = r.get("base_lps"), r.get("bias_lps")
    out = {"p_ratio_norm (shipped)": r.get("p_ratio_norm"),
           "p_ratio_signed": r.get("p_ratio_signed")}
    if not b or not x: return out
    for k in (4, 8, 16, 32, 64):
        mb, mx = _m(b, k), _m(x, k)
        if mb is not None and mx is not None:
            out[f"first{k}_abs"] = abs(mb - mx)
            out[f"first{k}_signed"] = mb - mx
    for how in ("median", "trim"):
        mb, mx = _m(b, None, how), _m(x, None, how)
        if mb is not None and mx is not None:
            out[f"full_{how}_abs"] = abs(mb - mx)
    return out


for path in paths:
    rows = []
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
            if "error" not in r: rows.append(r)
        except Exception: pass
    for r in rows: r["_v"] = variants(r)
    names = list(rows[0]["_v"])
    atk = [r for r in rows if r["prompt_type"] in ATK]
    ben = [r for r in rows if r["prompt_type"] == "benign"]
    bsrc = sorted({r.get("dataset") for r in ben if r.get("dataset")})

    print("\n" + "=" * 84)
    print(f"{os.path.basename(path)}   {len(atk)} attack / {len(ben)} benign")
    print("=" * 84)
    hdr = f"{'variant':<24}{'all benign':>12}" + "".join(f"{s[:12]:>14}" for s in bsrc)
    print(hdr + f"{'hard-easy':>11}")
    print("-" * len(hdr + f"{'hard-easy':>11}"))
    best = None
    for n in names:
        allv = orient(auc([r["_v"].get(n) for r in atk], [r["_v"].get(n) for r in ben]))
        if allv is None: continue
        cells, per = "", {}
        for s in bsrc:
            sub = [r for r in ben if r.get("dataset") == s]
            v = orient(auc([r["_v"].get(n) for r in atk], [r["_v"].get(n) for r in sub]))
            per[s] = v
            cells += f"{v:>14.3f}" if v is not None else f"{'-':>14}"
        d = ""
        if "alpaca" in per and "xstest_safe" in per and None not in (per["alpaca"], per["xstest_safe"]):
            d = f"{per['xstest_safe'] - per['alpaca']:>+11.3f}"
        mark = ""
        if best is None or allv > best[1]:
            best = (n, allv)
        print(f"{n:<24}{allv:>12.3f}{cells}{d}{mark}")
    if best:
        print(f"\n  best overall: {best[0]}  ({best[1]:.3f})   "
              f"shipped: {orient(auc([r['_v'].get(names[0]) for r in atk], [r['_v'].get(names[0]) for r in ben])):.3f}")
