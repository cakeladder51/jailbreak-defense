# -*- coding: utf-8 -*-
"""Box + strip plot of p_ratio_norm per dataset, one figure per model.

The aggregate attack-vs-benign number hides that the families behave very differently -- and
that one of them (cipherchat_cipher) sits on the wrong side entirely. This is the figure that
shows it: every prompt as a point, one column per dataset, attack families and the two benign
sources side by side.

Columns are coloured by prompt type, and the two benign sources are kept apart on purpose:
alpaca is the EASY benign set and xstest_safe the HARD one (prompts that look harmful but are
not). If a signal only separates attacks from alpaca and not from xstest_safe, it is detecting
unusual text rather than harm.

usage: python plot_ratio_by_dataset.py [<cgp.jsonl> ...]   (default: all *_hb_cgp.jsonl)
Writes docs/figures/<model>_<tag>_ratio_by_dataset.png (no prompt text -- safe to commit).
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(PROJ, "docs", "figures")
KEY = os.environ.get("CGP_PLOT_KEY", "p_ratio_norm")
YLABEL = "Normalized Log-Prob Ratio" if KEY == "p_ratio_norm" else KEY

# compile order, so the figure reads the same way as every table in docs/
ORDER = ["advbench", "advbench_prefill", "safemtdata_multiturn", "cipherchat_cipher",
         "artprompt_orthographic", "msj_contextwindow", "jailbreakbench_general",
         "alpaca", "xstest_safe"]
COLOR = {"harmful": "#4a5568", "forced_prefix": "#3d8f8f",
         "alpaca": "#7fc17f", "xstest_safe": "#d9a441"}
BOX_FILL = "#ebebeb"


def colour_for(ptype, dataset):
    if ptype == "benign":
        return COLOR.get(dataset, "#7fc17f")
    return COLOR.get(ptype, "#4a5568")


def plot(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
            if "error" not in r and r.get(KEY) is not None: rows.append(r)
        except Exception: pass
    if not rows:
        print("no usable rows in", path); return

    present = [d for d in ORDER if any(r.get("dataset") == d for r in rows)]
    present += sorted({r.get("dataset") for r in rows} - set(present) - {None})
    data, cols, ptypes = [], [], []
    for d in present:
        sub = [r for r in rows if r.get("dataset") == d]
        data.append([r[KEY] for r in sub])
        cols.append(d)
        ptypes.append(sub[0]["prompt_type"])

    n_ben = sum(1 for p in ptypes if p == "benign")
    fig, ax = plt.subplots(figsize=(max(11, 1.5 * len(cols) + 3), 7.6))

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", color="#c9c9c9", alpha=.7, linewidth=.9)
    ax.xaxis.grid(False)

    bp = ax.boxplot(data, positions=range(1, len(cols) + 1), widths=.58,
                    showfliers=False, patch_artist=True, whis=1.5)
    for b in bp["boxes"]:
        b.set(facecolor=BOX_FILL, edgecolor="#9a9a9a", linewidth=1.0, zorder=2)
    for part in ("whiskers", "caps"):
        for it in bp[part]: it.set(color="#3a3a3a", linewidth=1.2, zorder=2)
    for m in bp["medians"]:
        m.set(color="#111111", linewidth=1.8, zorder=3)

    # A couple of cipherchat rows sit 3-4x above everything else and flatten the rest of the
    # figure into an unreadable strip. Clip the axis to just above the tallest whisker and draw
    # the off-scale points as carets on the top edge, labelled -- visible, not silently dropped.
    whis_hi = []
    for v in data:
        if len(v) < 2: continue
        q1, q3 = np.percentile(v, [25, 75])
        inside = [x for x in v if x <= q3 + 1.5 * (q3 - q1)]
        whis_hi.append(max(inside) if inside else q3)
    ylim_hi = max(whis_hi) * 1.28 if whis_hi else 1.0

    rng = np.random.default_rng(67)
    n_off = 0
    for i, (vals, d, pt) in enumerate(zip(data, cols, ptypes), start=1):
        if not vals: continue
        v = np.asarray(vals, float)
        x = i + rng.uniform(-.19, .19, len(v))
        on, off = v <= ylim_hi, v > ylim_hi
        ax.scatter(x[on], v[on], s=58, alpha=.78, color=colour_for(pt, d),
                   edgecolor="white", linewidth=.6, zorder=4)
        if off.any():
            n_off += int(off.sum())
            ax.scatter(x[off], np.full(off.sum(), ylim_hi * .985), s=64, marker="^",
                       color=colour_for(pt, d), edgecolor="#333333", linewidth=.7,
                       zorder=5, clip_on=False)
            ax.annotate(f"{v[off].max():.2f}", (i, ylim_hi * .955), ha="center",
                        fontsize=8.5, color="#444444", zorder=6)
    ax.set_ylim(min(0.0, float(min(min(v) for v in data if v)) * 1.05), ylim_hi)

    # divider between attack families and the benign block
    # Group labels live just ABOVE the axes, so they cannot collide with the off-scale carets.
    if n_ben and n_ben < len(cols):
        ax.axvline(len(cols) - n_ben + .5, color="#9a9a9a", linewidth=1.1,
                   linestyle=(0, (5, 4)), zorder=1)
        trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text((len(cols) - n_ben + 1) / 2.0, 1.012, "attack families", transform=trans,
                ha="center", va="bottom", fontsize=10.5, color="#5a5a5a")
        ax.text(len(cols) - (n_ben - 1) / 2.0, 1.012, "benign", transform=trans,
                ha="center", va="bottom", fontsize=10.5, color="#5a5a5a")

    ax.set_xticks(range(1, len(cols) + 1))
    ax.set_xticklabels([f"{c}\n(n={len(v)})" for c, v in zip(cols, data)],
                       fontsize=9.5, rotation=28, ha="right")
    ax.set_xlabel("Dataset", fontsize=12, labelpad=8)
    ax.set_ylabel(YLABEL, fontsize=12)
    base = os.path.basename(path).replace("_cgp.jsonl", "")
    model = base.rsplit("_", 2)[0]
    subtitle = f"({len(rows)} samples)"
    if n_off:
        subtitle += f" — {n_off} point{'s' if n_off > 1 else ''} above axis, shown as ▲"
    ax.set_title(f"Distribution of Probability Ratios by dataset — {model}  {subtitle}",
                 fontsize=13.5, pad=26)
    for s in ax.spines.values():
        s.set_color("#4a4a4a"); s.set_linewidth(.9)
    ax.tick_params(labelsize=10)

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f"{base}_ratio_by_dataset.png")
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    med = {c: float(np.median(v)) for c, v in zip(cols, data) if v}
    print(f"wrote {out}")
    print("   medians: " + "  ".join(f"{c}={m:.3f}" for c, m in med.items()))


paths = sys.argv[1:] or sorted(glob.glob(os.path.join(PROJ, "cgp_out", "*_hb_cgp.jsonl")))
if not paths:
    sys.exit("no *_hb_cgp.jsonl found -- pass paths explicitly")
for p in paths:
    plot(p)
