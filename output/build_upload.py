"""Build the two root Jupyter upload files from the published source snapshot."""

from __future__ import annotations

import gzip
import io
import json
import os
import shutil
import tarfile
import tempfile
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / "nutrition_qaoa_result_public"
NOTEBOOK = ROOT / "start_here.ipynb"
ARCHIVE = ROOT / "nutrition_qaoa_extended.tar.gz"
SOURCE_FILES = (
    "README.md", "start_here.ipynb", "requirements.txt", "config.json",
    "model.py", "progress.py", "gpu_probe.py", "run_experiments.py",
    "fetch_usda.py", "build_bundle.py", "test_model.py",
    "test_experiment_design.py", "test_progress.py", "test_gpu_probe.py",
    "test_bundle_bootstrap.py", "data/foods.csv", "data/food_manifest.csv",
)

INTRODUCTION = """# Nutrition QAOA: revised experiment

Upload this notebook and `nutrition_qaoa_extended.tar.gz` to the same Jupyter
directory, choose a **Python 3.11 or 3.12 kernel**, and select **Run All**.
Request **4 CPUs and 16 GB RAM** for the default CPU run. The first run needs
internet access for package installation. The archive includes all food data.
The notebook reports GPU availability and uses CPU if no usable GPU is visible.

## Model and experiment

The archive contains the new logic: `model.py` defines the true nutrient cost,
the exact food-diversity reward, and the quadratic surrogate. `config.json`
contains the nutrient references and weights. `run_experiments.py` implements
QAOA, exhaustive comparison, and diagnostics. The next cell extracts these
files before running them.

The model has **no fixed total-serving target or penalty**. Eight foods have
0-3 portions each, encoded in one 16-qubit circuit. Diversity rewards each
selected food type once, independently of repeated portions. Brute force
scores all 65,536 menus with the true objective; QAOA optimizes the validated
quadratic surrogate, and measured menus are evaluated with the true objective.

The default comparison uses **p=1,2,3,4**, **seeds 1,7,21**, **3 restarts**, and
**COBYLA / Nelder-Mead**: 72 optimizations with a 400-objective-evaluation cap
per optimization. It also compares 1,024 / 4,096 / 16,384 measurements and runs
100 nutrition-policy sensitivity trials. Edit the experiment settings cell
before execution to change this design.

This notebook has no saved execution outputs. The separately published
`nutrition_qaoa_result_public/start_here.ipynb` preserves the completed CPU run,
with its full outputs under `nutrition_qaoa_result_public/outputs/runs/`.
"""


def build() -> None:
    missing = [name for name in SOURCE_FILES if not (PROJECT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing published source files: {missing}")
    notebook = json.loads((PROJECT / "start_here.ipynb").read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = INTRODUCTION.splitlines(keepends=True)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            cell["metadata"] = {}
            compile("".join(cell["source"]), "start_here.ipynb", "exec")
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    notebook_bytes = (json.dumps(notebook, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    NOTEBOOK.write_bytes(notebook_bytes)

    payloads = {
        name: notebook_bytes if name == "start_here.ipynb" else (PROJECT / name).read_bytes()
        for name in SOURCE_FILES
    }
    # Stable metadata avoids publishing local account names or workstation paths.
    with ARCHIVE.open("wb") as destination:
        with gzip.GzipFile(fileobj=destination, filename="", mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as bundle:
                for name, payload in payloads.items():
                    entry = tarfile.TarInfo(name)
                    entry.size = len(payload)
                    entry.mode = 0o644
                    bundle.addfile(entry, io.BytesIO(payload))

    with tarfile.open(ARCHIVE, "r:gz") as bundle:
        if set(bundle.getnames()) != set(SOURCE_FILES):
            raise ValueError("Upload archive has unexpected contents")
        for entry in bundle.getmembers():
            if entry.uid or entry.gid or entry.uname or entry.gname:
                raise ValueError("Upload archive contains account metadata")
            if bundle.extractfile(entry).read() != payloads[entry.name]:
                raise ValueError(f"Archive content mismatch: {entry.name}")

    # Test the actual two-file upload path without installing packages or running QAOA.
    previous_dir = Path.cwd()
    with tempfile.TemporaryDirectory() as temporary:
        upload_dir = Path(temporary)
        shutil.copyfile(ARCHIVE, upload_dir / ARCHIVE.name)
        try:
            os.chdir(upload_dir)
            scope = {}
            with redirect_stdout(io.StringIO()):
                exec(compile("".join(notebook["cells"][1]["source"]), "bootstrap", "exec"), scope)
            extracted = scope["project_dir"]
            for name in SOURCE_FILES:
                if (extracted / name).read_bytes() != payloads[name]:
                    raise ValueError(f"Notebook extracted incorrect content: {name}")
        finally:
            os.chdir(previous_dir)
    print(f"Built {NOTEBOOK.name} and {ARCHIVE.name} with {len(SOURCE_FILES)} source files")
    print("Verified notebook code, source-file contents, archive privacy metadata, and two-file bootstrap")


if __name__ == "__main__":
    build()
