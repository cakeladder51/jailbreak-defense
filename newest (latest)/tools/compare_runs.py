# -*- coding: utf-8 -*-
"""Compare judge-free reports across runs: A1 separation, controls, stability, cost.

usage: python compare_runs.py [outdir]
Reads cgp_out/<model>_<tag>_judgefree.json written by judge_free_report().

All AUCs are shown ORIENTED (max(a, 1-a)) with the direction named separately. Raw AUC puts
attacks as the positive class, so p_ratio_norm landing below 0.5 is the direction the method
predicts, not a failure -- fit_threshold learns that sign. Comparing raw values against 0.5
as if higher were better makes a working signal look broken.
"""
import json, glob, os, sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJ, "cgp_out")

reps = []
for p in sorted(glob.glob(os.path.join(OUT, "*_judgefree.json"))):
    try:
        reps.append((os.path.basename(p), json.load(open(p, encoding="utf-8"))))
    except Exception as e:
        print("skip", p, e)
if not reps:
    sys.exit("no *_judgefree.json in " + OUT)

KEYS = ["p_ratio_norm", "refusal_then_gap", "bias_is_refusal", "base_avg", "H"]
CONTROLS = ("base_avg", "H")


def ent(r, k):
    return ((r.get("separation") or {}).get(k) or {})


def orient(e):
    """Oriented AUC, preferring the value the notebook stored, else derived."""
    if e.get("AUC_oriented") is not None: return e["AUC_oriented"]
    a = e.get("AUC")
    return None if a is None else max(a, 1.0 - a)


def direction(e):
    if e.get("direction"): return e["direction"]
    a = e.get("AUC")
    return None if a is None else ("attacks-lower" if a < 0.5 else "attacks-higher")


def f(v, w=6, d=3):
    return ("{:%d.%df}" % (w, d)).format(v) if isinstance(v, (int, float)) else "{:>{w}}".format("-", w=w)


print("\n" + "=" * 100)
print("A1 -- ORIENTED SEPARATION AUC (attack vs benign).  0.50 = no separation.")
print("     p_ratio_norm is the shipped signal; base_avg and H are the controls it must BEAT.")
print("     refusal_then_gap / bias_is_refusal are candidates (present only in newer runs).")
print("=" * 100)
hdr = "{:<30}".format("run") + "".join("{:>17}".format(k[:16]) for k in KEYS) + "{:>10}".format("n_atk/ben")
print(hdr); print("-" * len(hdr))
for _, r in reps:
    tag = f"{r.get('model')} [{r.get('run_tag')}]"
    row = "{:<30}".format(tag[:29])
    for k in KEYS:
        row += "{:>17}".format(f(orient(ent(r, k))))
    h = ent(r, "p_ratio_norm")
    row += "{:>10}".format(f"{h.get('n_attack','?')}/{h.get('n_benign','?')}")
    print(row)

print("\n  headline with 95% CI, direction, and whether a control matches it:")
for _, r in reps:
    e = ent(r, "p_ratio_norm")
    if e.get("AUC") is None: continue
    lo, hi = e.get("lo"), e.get("hi")
    ci = f"[{min(lo,1-lo) if lo else 0:.3f}, {max(hi,1-hi) if hi else 0:.3f}]" if lo is not None else "-"
    beat = ""
    for c in CONTROLS:
        ce = ent(r, c)
        if orient(ce) is not None and orient(ce) >= orient(e):
            beat += f"  !! {c} ({orient(ce):.3f}) >= headline"
    print(f"    {r.get('model')} [{r.get('run_tag')}]: oriented={orient(e):.3f} "
          f"raw={e['AUC']:.3f} ({direction(e)}){beat}")

# ---- easy vs hard benign ------------------------------------------------------------
if any(ent(r, "p_ratio_norm").get("by_benign") for _, r in reps):
    print("\n" + "=" * 100)
    print("EASY vs HARD BENIGN -- oriented AUC of attacks against each benign source.")
    print("     alpaca = easy (no surface resemblance to attacks); xstest_safe = hard (looks")
    print("     harmful, is not). A detector of 'unusual text' collapses in the hard column.")
    print("=" * 100)
    for _, r in reps:
        bb = {k: ent(r, k).get("by_benign") or {} for k in KEYS}
        srcs = sorted({s for v in bb.values() for s in v})
        if not srcs: continue
        print(f"\n  {r.get('model')} [{r.get('run_tag')}]")
        print("    " + "{:<19}".format("signal") + "".join("{:>16}".format(s[:15]) for s in srcs)
              + "{:>9}".format("drop"))
        for k in KEYS:
            if not bb[k]: continue
            vals = [max(bb[k][s], 1 - bb[k][s]) if bb[k].get(s) is not None else None for s in srcs]
            if any(v is None for v in vals): continue
            easy = vals[srcs.index("alpaca")] if "alpaca" in srcs else max(vals)
            hard = vals[srcs.index("xstest_safe")] if "xstest_safe" in srcs else min(vals)
            print("    " + "{:<19}".format(k) + "".join("{:>16.3f}".format(v) for v in vals)
                  + "{:>+9.3f}".format(hard - easy))

# ---- per family ---------------------------------------------------------------------
print("\n" + "=" * 100)
print("PER-FAMILY ORIENTED AUC (p_ratio_norm) -- consistent across runs?")
print("=" * 100)
fams = sorted({d for _, r in reps for d in ent(r, "p_ratio_norm").get("by_dataset", {})})
if fams:
    hdr = "{:<28}".format("family") + "".join(
        "{:>22}".format(f"{r.get('model')[:9]}/{r.get('run_tag')[:10]}") for _, r in reps)
    print(hdr); print("-" * len(hdr))
    cols = []
    for ds in fams:
        line = "{:<28}".format(ds[:27]); vals = []
        for _, r in reps:
            v = (ent(r, "p_ratio_norm").get("by_dataset") or {}).get(ds)
            v = None if v is None else max(v, 1 - v)
            vals.append(v); line += "{:>22}".format(f(v))
        cols.append((ds, vals)); print(line)
    if len(reps) >= 2:
        def rank(xs):
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            rk = [0]*len(xs)
            for pos, i in enumerate(order): rk[i] = pos
            return rk
        for a in range(len(reps)):
            for b in range(a+1, len(reps)):
                pairs = [(v[a], v[b]) for _, v in cols if v[a] is not None and v[b] is not None]
                if len(pairs) < 3: continue
                xa, xb = rank([p[0] for p in pairs]), rank([p[1] for p in pairs])
                n = len(pairs); d2 = sum((xa[i]-xb[i])**2 for i in range(n))
                print(f"  Spearman(family ranking)  {reps[a][1]['model']}/{reps[a][1]['run_tag']}"
                      f"  vs  {reps[b][1]['model']}/{reps[b][1]['run_tag']}:  "
                      f"rho={1 - 6*d2/(n*(n*n-1)):+.3f}  (n={n})")

print("\n" + "=" * 100)
print("THRESHOLD STABILITY  &  COST")
print("=" * 100)
for _, r in reps:
    st, c = r.get("stability") or {}, r.get("cost") or {}
    print(f"\n  {r.get('model')} [{r.get('run_tag')}]")
    if st:
        pf = st.get("direction_pos_frac", 0.0)
        # Naming the direction matters: a stable "attacks-lower" fit is the method working
        # as designed. Only a sign that CHANGES between splits is a defect.
        if pf >= 0.95:   d = "stable (attacks-higher)"
        elif pf <= 0.05: d = "stable (attacks-lower -- the predicted direction)"
        else:            d = f"UNSTABLE ({pf*100:.0f}% attacks-higher) <-- fitted sign changes between splits"
        print(f"    threshold  {st['thr_mean']:.4f} +/- {st['thr_sd']:.4f}   "
              f"(p05-p95 {st['thr_p05']:.4f} .. {st['thr_p95']:.4f}) over {st['n_repeats']} splits")
        print(f"    direction  {d}")
    if c:
        if c.get("cgp_overhead_x"):
            print(f"    cost       base {c['t_base_s']:.2f}s + bias {c['t_bias_s']:.2f}s "
                  f"= {c['cgp_overhead_x']:.2f}x" + (f"   (safedecoding {c['t_sd_s']:.2f}s)"
                                                     if c.get("t_sd_s") else ""))
        if c.get("proc_fired_frac") is not None:
            print(f"    processor  fired on {c['proc_fired_frac']*100:.0f}% of rows, "
                  f"mean {c.get('proc_applied_mean') or 0:.1f} biased steps")
print()
