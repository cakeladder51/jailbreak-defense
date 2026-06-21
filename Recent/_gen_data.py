import json

nb = json.load(open("CGP - .ipynb", encoding="utf-8"))
# cell 0 is the dataset-compilation script
src = "".join(nb["cells"][0]["source"])
exec(compile(src, "cell0", "exec"))
