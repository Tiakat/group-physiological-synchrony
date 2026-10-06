#!/usr/bin/env python3
"""Execute a notebook's code cells top-to-bottom (CI-friendly, no Jupyter needed).

Strips IPython magics, forces the Agg backend, runs each code cell in the
notebook's directory, and fails loudly on the first error. Used to verify
that notebooks/01_eda_synchrony.ipynb reproduces end to end.
"""
import json
import sys
import traceback
from pathlib import Path

nb_path = Path(sys.argv[1])
nb = json.loads(nb_path.read_text())
ns = {"__name__": "__main__"}

# headless before any cell runs
import matplotlib
matplotlib.use("Agg")

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"{len(code_cells)} code cells in {nb_path.name}")

import os
os.chdir(nb_path.parent)  # cells use relative paths like ../src, ../data

for i, cell in enumerate(code_cells):
    src = "".join(cell["source"])
    src = "\n".join(l for l in src.splitlines() if not l.strip().startswith("%"))
    print(f"--- cell {i + 1}/{len(code_cells)} ---")
    try:
        exec(compile(src, f"<cell {i + 1}>", "exec"), ns)
    except Exception:
        print(f"CELL {i + 1} FAILED")
        traceback.print_exc()
        sys.exit(1)
print("ALL CELLS OK")
