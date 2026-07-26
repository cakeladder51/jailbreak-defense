# -*- coding: utf-8 -*-
"""Compare judge-free reports across runs: A1 separation, controls, stability, cost.

usage: python compare_runs.py [outdir]
Reads cgp_out/<model>_<tag>_judgefree.json written by judge_free_report().
"""
import json, glob, os, sys
from collections import defaultdict

OUT = sys.argv[1] if len(sys.argv) > 1 else r"D:\CodeFiles\AI project\newest (latest)\cgp_out"

reps = []
for p in sorted(glob.glob(os.path.join(OUT, "*_judgefree.json"))):
    try:
        reps.append((os.path.basename(p), json.load(open(p, encoding="utf-8"))))
    except Exception as e:
        print("skip", p, e)
if not reps:
    sys.exit("no *_judgefree.json in " + OUT)

def f(v, w=6, d=3):
    return ("{:%d.%df}" % (w, d)).format(v) if isinstance(v, (int, float)) else "{:>{w}}".format("-", w=w)

KEYS = ["p_ratio_norm", "p_ratio_signed", "base_avg", "bias_avg", "H"]

print("\n" + "=" * 100)
print("A1 -- SEPARATION AUC (attack vs benign).  0.50 = no separation.")
print("     'headline' is p_ratio_norm; base_avg and H are the controls that must be BEATEN.")
print("=" * 100)
hdr = "{:<34}".format("run") + "".join("{:>13}".format(k[:12]) for k in KEYS) + "{:>10}".format("n_atk/ben")
print(hdr); print("-" * len(hdr))
for name, r in reps:
    sep = r.get("separation", {})
    tag = f"{r.get('model')} [{r.get('run_tag')}]"
    row = "{:<34}".format(tag[:33])
    for k in KEYS:
        e = sep.get(k) or {}
        row += "{:>13}".format(f(e.get("AUC")) if e.get("AUC") is not None else "-")
    h = sep.get("p_ratio_norm") or {}
    row += "{:>10}".format(f"{h.get('n_attack','?')}/{h.get('n_benign','?')}")
    print(row)

print("\n  with 95% CI on the headline signal:")
for name, r in reps:
    e = (r.get("separation") or {}).get("p_ratio_norm") or {}
    if e.get("AUC") is None: continue
    ci = f"[{e['lo']:.3f}, {e['hi']:.3f}]" if e.get("lo") is not None else "-"
    beat = ""
    for ctrl in ("base_avg", "H"):
        c = (r.get("separation") or {}).get(ctrl) or {}
        if c.get("AUC") is not None:
            # both signals may point either way; compare distance from chance
            if abs(c["AUC"] - .5) >= abs(e["AUC"] - .5):
                beat += f"  !! {ctrl} (AUC {c['AUC']:.3f}) separates >= headline"
    print(f"    {r.get('model')} [{r.get('run_tag')}]: AUC={e['AUC']:.3f} {ci} "
          f"delta={e.get('cliffs_delta'):.3f}{beat}")

print("\n" + "=" * 100)
print("PER-FAMILY AUC (headline signal) -- is the ranking of attack families consistent across runs?")
print("=" * 100)
fams = sorted({d for _, r in reps
               for d in ((r.get("separation") or {}).get("p_ratio_norm") or {}).get("by_dataset", {})})
if fams:
    hdr = "{:<28}".format("family") + "".join("{:>22}".format(f"{r.get('model')[:9]}/{r.get('run_tag')[:10]}")
                                              for _, r in reps)
    print(hdr); print("-" * len(hdr))
    cols = []
    for ds in fams:
        line = "{:<28}".format(ds[:27]); vals = []
        for _, r in reps:
            v = (((r.get("separation") or {}).get("p_ratio_norm") or {}).get("by_dataset") or {}).get(ds)
            vals.append(v); line += "{:>22}".format(f(v) if v is not None else "-")
        cols.append((ds, vals)); print(line)
    # Spearman between the first two runs, if both present
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
                n = len(pairs)
                d2 = sum((xa[i]-xb[i])**2 for i in range(n))
                rho = 1 - 6*d2/(n*(n*n-1))
                print(f"  Spearman(family ranking)  {reps[a][1]['model']}/{reps[a][1]['run_tag']}"
                      f"  vs  {reps[b][1]['model']}/{reps[b][1]['run_tag']}:  rho={rho:+.3f}  (n={n})")

print("\n" + "=" * 100)
print("THRESHOLD STABILITY  &  COST")
print("=" * 100)
for name, r in reps:
    st, c = r.get("stability") or {}, r.get("cost") or {}
    tag = f"{r.get('model')} [{r.get('run_tag')}]"
    print(f"\n  {tag}")
    if st:
        flip = ("STABLE" if st.get("direction_stable") else
                f"FLIPS ({st.get('direction_pos_frac',0)*100:.0f}% positive)  <-- routing inverts")
        print(f"    threshold  {st['thr_mean']:.4f} +/- {st['thr_sd']:.4f}   "
              f"(p05-p95 {st['thr_p05']:.4f} .. {st['thr_p95']:.4f}) over {st['n_repeats']} splits")
        print(f"    direction  {flip}")
    if c:
        if c.get("cgp_overhead_x"):
            print(f"    cost       base {c['t_base_s']:.2f}s + bias {c['t_bias_s']:.2f}s "
                  f"= {c['cgp_overhead_x']:.2f}x" + (f"   (safedecoding {c['t_sd_s']:.2f}s)"
                                                     if c.get("t_sd_s") else ""))
        if c.get("proc_fired_frac") is not None:
            print(f"    processor  fired on {c['proc_fired_frac']*100:.0f}% of rows, "
                  f"mean {c.get('proc_applied_mean') or 0:.1f} biased steps, "
                  f"stop@{c.get('proc_stop_step_mean')}")
print()
