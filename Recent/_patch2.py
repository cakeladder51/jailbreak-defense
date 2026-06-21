import json

NB = "CGP - .ipynb"
RUNNER_ID = "a76f624f-33c8-4397-9390-abc786273c82"
LAUNCH_ID = "e84b85c0-6cc8-493d-b73c-19a3a7fe889a"
PLOT_ID = "plot-ratio-dist"

nb = json.load(open(NB, encoding="utf-8"))


def cell(cid):
    return next(c for c in nb["cells"] if c.get("id") == cid)


def patch(cid, repls):
    c = cell(cid)
    s = "".join(c["source"])
    for old, new, n in repls:
        assert s.count(old) == n, f"[{cid}] {old!r} found {s.count(old)} != {n}"
        s = s.replace(old, new)
    c["source"] = s.splitlines(keepends=True)


patch(RUNNER_ID, [
    # bf16 dtype in model loader
    ("            model_id,\n            token=HF_TOKEN,\n            device_map={\"\": torch.cuda.current_device()},",
     "            model_id,\n            token=HF_TOKEN,\n            torch_dtype=DTYPE,\n            device_map={\"\": torch.cuda.current_device()},", 1),
    # enable KV cache for generation (perf; gen dict only)
    ("\"use_cache\": False,", "\"use_cache\": True,", 1),
])

patch(LAUNCH_ID, [
    ("    \"Llama-3.1-8B\": \"meta-llama/Meta-Llama-3.1-8B-Instruct\",",
     "    \"Llama-3.2-3B\": \"meta-llama/Llama-3.2-3B-Instruct\",", 1),
])

patch(PLOT_ID, [
    ("results_parallel/results_Llama-3.1-8B.csv",
     "results_parallel/results_Llama-3.2-3B.csv", 1),
])

json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
open(NB, "a", encoding="utf-8").write("\n")

# also emit parallel_runner.py from the (now patched) runner cell, minus the magic line
src = "".join(cell(RUNNER_ID)["source"])
src = "\n".join(l for l in src.splitlines() if not l.strip().startswith("%%"))
open("parallel_runner.py", "w", encoding="utf-8").write(src + "\n")
print("OK: patched + wrote parallel_runner.py")
