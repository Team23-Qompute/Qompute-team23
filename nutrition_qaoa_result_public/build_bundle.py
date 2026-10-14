"""Create a Jupyter upload folder with exactly one notebook and one tar.gz."""

from pathlib import Path
import shutil
import tarfile


PROJECT = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT.parent / "nutrition_qaoa_upload"
ARCHIVE = UPLOAD_DIR / "nutrition_qaoa_extended.tar.gz"
NOTEBOOK = UPLOAD_DIR / "start_here.ipynb"
SOURCE_FILES = (
    "README.md",
    "start_here.ipynb",
    "requirements.txt",
    "config.json",
    "model.py",
    "progress.py",
    "gpu_probe.py",
    "run_experiments.py",
    "fetch_usda.py",
    "build_bundle.py",
    "test_model.py",
    "test_experiment_design.py",
    "test_progress.py",
    "test_gpu_probe.py",
    "test_bundle_bootstrap.py",
    "data/foods.csv",
    "data/food_manifest.csv",
)


def main() -> None:
    missing = [name for name in SOURCE_FILES if not (PROJECT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot build bundle; missing: {missing}")
    UPLOAD_DIR.mkdir(exist_ok=True)
    unexpected = sorted(path.name for path in UPLOAD_DIR.iterdir() if path.name not in {ARCHIVE.name, NOTEBOOK.name})
    if unexpected:
        raise RuntimeError(f"Upload folder contains unrelated files; preserving them: {unexpected}")
    with tarfile.open(ARCHIVE, "w:gz") as bundle:
        for name in SOURCE_FILES:
            bundle.add(PROJECT / name, arcname=name, recursive=False)
    with tarfile.open(ARCHIVE, "r:gz") as bundle:
        bundled_names = {entry.name for entry in bundle.getmembers() if entry.isfile()}
    if bundled_names != set(SOURCE_FILES):
        raise RuntimeError("Archive content verification failed")
    shutil.copy2(PROJECT / "start_here.ipynb", NOTEBOOK)
    print(f"Created {UPLOAD_DIR}")
    print(f"  {NOTEBOOK.name} ({NOTEBOOK.stat().st_size:,} bytes)")
    print(f"  {ARCHIVE.name} ({ARCHIVE.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
