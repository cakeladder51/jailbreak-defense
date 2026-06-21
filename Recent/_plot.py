import json
nb = json.load(open("CGP - .ipynb", encoding="utf-8"))
src = "".join(next(c for c in nb["cells"] if c.get("id") == "plot-ratio-dist")["source"])
import matplotlib
matplotlib.use("Agg")
exec(compile(src, "plot-cell", "exec"))
