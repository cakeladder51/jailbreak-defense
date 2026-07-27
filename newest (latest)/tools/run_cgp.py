# -*- coding: utf-8 -*-
"""Drive the CGP notebook for ONE model at ONE precision.

usage: python run_cgp.py <run_tag> <model_key> <quant>
Env overrides: CGP_N_PER_DATASET, CGP_N_BENIGN
"""
import json, os, sys, traceback

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJ)

TAG, MODEL_KEY, QUANT = sys.argv[1], sys.argv[2], sys.argv[3]

nb = json.load(open("CGP (1).ipynb", encoding="utf-8"))
S = lambda i: "".join(nb["cells"][i]["source"])

g = {"__name__": "__main__"}
exec(compile(S(2), "<config>", "exec"), g)          # cell 2: config (MODE="real")

mid = g["MODELS"][MODEL_KEY]
g["RUN_TAG"]    = TAG
g["MODELS"]     = {MODEL_KEY: mid}
# Keep the model's other MODEL_LOAD flags (e.g. trust_remote_code=False for phi-3-mini)
# and override only the precision. Replacing the whole dict with {"quant": QUANT} silently
# dropped those flags -- phi-3-mini then loaded the stale hub modeling_phi3.py and died on
# KeyError 'type', even though the same model loaded fine outside this driver.
load = dict((g.get("MODEL_LOAD") or {}).get(MODEL_KEY) or {})
load["quant"] = QUANT
g["MODEL_LOAD"] = {MODEL_KEY: load}
print(f"=== RUN {TAG}: {MODEL_KEY} ({mid}) quant={QUANT} "
      f"n_per_ds={g['N_PER_DATASET']} n_benign={g['N_BENIGN']} ===", flush=True)

try:
    exec(compile(S(3), "<core>", "exec"), g)        # cell 3: everything
    exec(compile(S(5), "<main>", "exec"), g)        # cell 5: main() -> runs it
except Exception:
    traceback.print_exc()
    sys.exit(f"RUN {TAG} FAILED")
print(f"=== RUN {TAG} COMPLETE ===", flush=True)
