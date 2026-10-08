"""Convierte hdt6.py (celdas separadas por '# %%') en hdt6.ipynb y lo ejecuta.

Uso:  python make_notebook.py
"""
import re

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

src = open("hdt6.py", encoding="utf-8").read()
chunks = re.split(r"^# %%(?: \[markdown\])?\n", src, flags=re.M)
heads = re.findall(r"^# %%( \[markdown\])?\n", src, flags=re.M)
nb = nbformat.v4.new_notebook()
for head, body in zip(heads, chunks[1:]):
    body = body.rstrip("\n")
    if head:   # celda markdown: quitar el prefijo '# '
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        nb.cells.append(nbformat.v4.new_markdown_cell(text))
    else:
        nb.cells.append(nbformat.v4.new_code_cell(body))
ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": "."}})
nbformat.write(nb, "hdt6.ipynb")
print("hdt6.ipynb generado con", len(nb.cells), "celdas")
