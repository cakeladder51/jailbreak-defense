# -*- coding: utf-8 -*-
"""Did the biased path actually produce DIFFERENT text from the baseline path?
If not, p_ratio_norm ~ 0 is an artifact of the experiment, not a property of CGP."""
import json, sys
import numpy as np

rows = []
for line in open(sys.argv[1], encoding="utf-8"):
    try:
        r = json.loads(line)
        if "error" not in r: rows.append(r)
    except Exception: pass

same = [r for r in rows if (r.get("base_text") or "") == (r.get("bias_text") or "")]
print(f"rows: {len(rows)}   base_text == bias_text: {len(same)}  ({100*len(same)/len(rows):.0f}%)")

by_ds = {}
for r in rows:
    k = r.get("dataset", "?")
    by_ds.setdefault(k, [0, 0])
    by_ds[k][1] += 1
    if (r.get("base_text") or "") == (r.get("bias_text") or ""): by_ds[k][0] += 1
print("\nidentical-output rate per family:")
for k, (s, n) in sorted(by_ds.items()):
    print(f"  {k:26s} {s:3d}/{n:3d}   {100*s/n:5.0f}%")

print("\nrefusal-string presence (crude, string match only -- NOT a judge):")
REF = ["i cannot", "i am unable", "as an ai", "i must", "i'm sorry", "i apologize"]
for field in ("base_text", "bias_text"):
    hit = sum(1 for r in rows if any(p in (r.get(field) or "").lower() for p in REF))
    print(f"  {field:10s} contains a refusal string: {hit:3d}/{len(rows)}")

print("\nexample where the two paths differ:")
diff = [r for r in rows if (r.get("base_text") or "") != (r.get("bias_text") or "")]
if diff:
    r = diff[0]
    print(f"  dataset={r.get('dataset')} type={r.get('prompt_type')} "
          f"p_ratio_norm={r.get('p_ratio_norm'):.4f}")
    print("  BASE:", repr((r.get("base_text") or "")[:180]))
    print("  BIAS:", repr((r.get("bias_text") or "")[:180]))
print("\nexample where they are IDENTICAL:")
if same:
    r = same[0]
    print(f"  dataset={r.get('dataset')} type={r.get('prompt_type')} "
          f"proc_applied={r.get('proc_applied')} proc_stop_step={r.get('proc_stop_step')}")
    print("  TEXT:", repr((r.get("base_text") or "")[:180]))
