import json
NB = "CGP - .ipynb"
nb = json.load(open(NB, encoding="utf-8"))


def cell(cid):
    return next(c for c in nb["cells"] if c.get("id") == cid)


def patch(cid, old, new):
    c = cell(cid)
    s = "".join(c["source"])
    assert s.count(old) == 1, f"{old!r} found {s.count(old)}"
    c["source"] = s.replace(old, new).splitlines(keepends=True)


patch("e84b85c0-6cc8-493d-b73c-19a3a7fe889a",
      "    \"Llama-3.2-3B\": \"meta-llama/Llama-3.2-3B-Instruct\",",
      "    \"Qwen2.5-3B\": \"Qwen/Qwen2.5-3B-Instruct\",")
patch("plot-ratio-dist",
      "results_parallel/results_Llama-3.2-3B.csv",
      "results_parallel/results_Qwen2.5-3B.csv")

json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
open(NB, "a", encoding="utf-8").write("\n")
print("OK: notebook -> Qwen2.5-3B")
