"""Create a privacy-safe, reproducible snapshot of the completed QAOA run.

The original downloaded results are never modified. The published CSV files
must remain byte-for-byte identical to their originals.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "nutrition_qaoa_result"
EXTRACTED = RAW / "extracted_ebc7c3a2268f"
PUBLISHED = ROOT / "nutrition_qaoa_result_public"

TEXT_SUFFIXES = {".csv", ".ipynb", ".json", ".jsonl", ".log", ".md", ".py", ".txt"}
BINARY_SUFFIXES = {".png"}


def replacements(text: str, account: str | None, hostname: str | None) -> str:
    text = re.sub(r"/home\d+/[^/\s\"'\\]+", "/home/REDACTED_USER", text)
    text = re.sub(r"[A-Za-z]:\\\\Users\\\\[^\\\\\s\"']+", r"C:\\Users\\REDACTED_USER", text)
    if account:
        text = text.replace(account, "REDACTED_SLURM_ACCOUNT")
    if hostname:
        text = text.replace(hostname, "REDACTED_CLUSTER_HOST")
    text = re.sub(r"\bJob \d+\b", "Job REDACTED_ID", text)
    text = re.sub(r"\bon nodes [A-Za-z0-9_-]+\b", "on nodes REDACTED_NODE", text)
    return text


def check_csv_unchanged(source: Path, target: Path) -> None:
    if source.read_bytes() != target.read_bytes():
        raise ValueError(f"CSV changed: {target}")
    with source.open(newline="", encoding="utf-8") as src, target.open(
        newline="", encoding="utf-8"
    ) as dst:
        if list(csv.reader(src)) != list(csv.reader(dst)):
            raise ValueError(f"CSV rows changed: {target}")


def numeric_tree(value):
    if isinstance(value, dict):
        return {key: numeric_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [numeric_tree(item) for item in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def main() -> None:
    if not (RAW / "start_here.ipynb").is_file() or not EXTRACTED.is_dir():
        raise FileNotFoundError("Expected downloaded notebook and extracted run are missing")
    if PUBLISHED.exists():
        raise FileExistsError(f"Refusing to overwrite {PUBLISHED}")

    run_dir = next((EXTRACTED / "outputs" / "runs").iterdir())
    account = json.loads((run_dir / "requested_settings.json").read_text(encoding="utf-8")).get(
        "slurm_account"
    )
    hostname = json.loads((run_dir / "gpu_probe.json").read_text(encoding="utf-8")).get(
        "hostname"
    )

    sources = [path for path in EXTRACTED.rglob("*") if path.is_file()]
    sources = [
        path
        for path in sources
        if "__pycache__" not in path.parts
        and path.name not in {"start_here.ipynb", ".bundle_sha256"}
        and path.suffix.lower() in TEXT_SUFFIXES | BINARY_SUFFIXES
    ]
    sources.append(RAW / "start_here.ipynb")

    counts = {"text": 0, "binary": 0, "csv": 0, "json": 0, "jsonl": 0}
    for source in sources:
        relative = Path("start_here.ipynb") if source == RAW / "start_here.ipynb" else source.relative_to(EXTRACTED)
        target = PUBLISHED / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() in BINARY_SUFFIXES:
            shutil.copyfile(source, target)
            counts["binary"] += 1
            continue
        original = source.read_text(encoding="utf-8")
        sanitized = replacements(original, account, hostname)
        if relative == Path("requirements.txt"):
            sanitized = sanitized.rstrip("\r\n") + "\n"
        if source.suffix.lower() == ".csv":
            if sanitized != original:
                raise ValueError(f"CSV contains private environment metadata: {source}")
            shutil.copyfile(source, target)
        else:
            target.write_text(sanitized, encoding="utf-8", newline="")
        counts["text"] += 1
        if source.suffix.lower() == ".csv":
            check_csv_unchanged(source, target)
            counts["csv"] += 1
        elif source.suffix.lower() in {".json", ".ipynb"}:
            original_obj = json.loads(original)
            sanitized_obj = json.loads(sanitized)
            if numeric_tree(original_obj) != numeric_tree(sanitized_obj):
                raise ValueError(f"Numeric JSON data changed: {target}")
            counts["json"] += 1
        elif source.suffix.lower() == ".jsonl":
            original_rows = [json.loads(line) for line in original.splitlines() if line]
            sanitized_rows = [json.loads(line) for line in sanitized.splitlines() if line]
            if numeric_tree(original_rows) != numeric_tree(sanitized_rows):
                raise ValueError(f"Numeric JSONL data changed: {target}")
            counts["jsonl"] += 1

    notebook = json.loads((PUBLISHED / "start_here.ipynb").read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4 or not notebook.get("cells"):
        raise ValueError("Published notebook did not validate")
    print(f"Published {len(sources)} files to {PUBLISHED}")
    print(
        f"Validated {counts['csv']} identical CSVs, {counts['json']} numeric JSON/notebook "
        f"files, and {counts['jsonl']} numeric JSONL files"
    )


if __name__ == "__main__":
    main()
