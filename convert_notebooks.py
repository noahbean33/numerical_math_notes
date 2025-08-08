#!/usr/bin/env python3
"""
convert_notebooks.py

Converts all Jupyter notebooks (.ipynb) in the current folder and subfolders into
Python scripts (.py).
- Extracts code cells into a .py file with the same basename, saved alongside the notebook.
- Preserves markdown cells as commented lines for context.
- Adds simple cell boundary markers and execution counts as comments.

Usage:
    python convert_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
import sys


def convert_ipynb_to_py(nb_path: Path) -> Path:
    """Convert a single .ipynb file to a .py file next to it.

    Returns the path to the created .py file.
    """
    with nb_path.open("r", encoding="utf-8") as f:
        try:
            nb = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"{nb_path.name} is not valid JSON: {e}") from e

    cells = nb.get("cells", [])

    py_lines: list[str] = []
    header = [
        "# -*- coding: utf-8 -*-\n",
        f"# Auto-generated from '{nb_path.name}' on {datetime.now().isoformat(timespec='seconds')}\n",
        "# Source: code cells extracted; markdown preserved as comments.\n",
        "\n",
    ]
    py_lines.extend(header)

    for idx, cell in enumerate(cells, start=1):
        cell_type = cell.get("cell_type")
        if cell_type == "code":
            exec_count = cell.get("execution_count")
            marker = f"# In [{exec_count}]" if exec_count is not None else f"# In [ ]"
            py_lines.append(marker + "\n")
            source = cell.get("source", [])
            # Ensure each line ends with a newline
            for line in source:
                if not line.endswith("\n"):
                    line = line + "\n"
                py_lines.append(line)
            # Ensure a blank line between cells
            if not py_lines or py_lines[-1].strip():
                py_lines.append("\n")
        elif cell_type == "markdown":
            py_lines.append("# %% [markdown]\n")
            for line in cell.get("source", []):
                if not line.endswith("\n"):
                    line = line + "\n"
                # Comment markdown lines
                py_lines.append("# " + line)
            if not py_lines or py_lines[-1].strip():
                py_lines.append("\n")
        else:
            # Unknown cell types are ignored but noted
            py_lines.append(f"# %% [ignored cell type: {cell_type}]\n\n")

    out_path = nb_path.with_suffix(".py")
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(py_lines)

    return out_path


def main() -> int:
    cwd = Path.cwd()
    search_root = cwd
    ipynb_files = [
        p for p in search_root.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in str(p)
    ]

    if not ipynb_files:
        print("No .ipynb files found in this folder or its subfolders.")
        return 0

    converted = []
    failed = []
    for nb_path in ipynb_files:
        try:
            out = convert_ipynb_to_py(nb_path)
            converted.append((nb_path.name, out.name))
        except Exception as e:
            failed.append((nb_path.name, str(e)))

    if converted:
        print("Converted notebooks:")
        for src, dst in converted:
            print(f" - {src} -> {dst}")
    if failed:
        print("\nFailed conversions:")
        for src, err in failed:
            print(f" - {src}: {err}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
