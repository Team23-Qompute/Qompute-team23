"""Fetch the eight source records from USDA FoodData Central.

This is an optional provenance check. It writes a candidate table and the raw
API responses; it does not overwrite the curated foods.csv used by the model.
Set FDC_API_KEY before running. See https://fdc.nal.usda.gov/api-guide/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import requests

from model import BASE_DIR, NUTRIENTS


# USDA FoodData Central nutrient IDs. Energy may be expressed using one of the
# alternative energy definitions; each retrieved record should be reviewed.
NUTRIENT_IDS = {
    "kcal": (1008, 2047, 2048),
    "protein_g": (1003,),
    "carbs_g": (1005,),
    "fat_g": (1004,),
    "potassium_mg": (1092,),
    "sodium_mg": (1093,),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=BASE_DIR / "data" / "food_manifest.csv"
    )
    parser.add_argument("--output", type=Path, default=BASE_DIR / "data" / "usda_download")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def nutrient_map(record: dict) -> dict[int, float]:
    values = {}
    for entry in record.get("foodNutrients", []):
        nutrient = entry.get("nutrient", {})
        nutrient_id = nutrient.get("id") or entry.get("nutrientId")
        amount = entry.get("amount")
        if nutrient_id is not None and amount is not None:
            values[int(nutrient_id)] = float(amount)
    return values


def candidate_row(record: dict, food: str, serving_g: float, fdc_id: int) -> dict:
    values = nutrient_map(record)
    row = {
        "food": food,
        "serving_g": serving_g,
        "fdc_id": fdc_id,
        "description": record.get("description", ""),
        "data_type": record.get("dataType", ""),
        "source_url": (
            "https://fdc.nal.usda.gov/fdc-app.html#/food-details/"
            f"{fdc_id}/nutrients"
        ),
    }
    for name in NUTRIENTS:
        candidates = NUTRIENT_IDS[name]
        selected_id = next((key for key in candidates if key in values), None)
        row[name] = (
            round(values[selected_id] * serving_g / 100.0, 4)
            if selected_id is not None
            else ""
        )
        row[f"{name}_nutrient_id"] = selected_id or ""
    return row


def main():
    args = parse_args()
    api_key = os.environ.get("FDC_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set the FDC_API_KEY environment variable using a USDA FoodData "
            "Central API key. The key is never written to output files."
        )

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    if not manifest:
        raise SystemExit("Manifest is empty")

    args.output.mkdir(parents=True, exist_ok=True)
    candidate_rows = []
    session = requests.Session()
    for source in manifest:
        fdc_id = int(source["fdc_id"])
        url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"
        response = session.get(
            url, params={"api_key": api_key}, timeout=args.timeout
        )
        response.raise_for_status()
        record = response.json()
        (args.output / f"fdc_{fdc_id}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        row = candidate_row(
            record, source["food"], float(source["serving_g"]), fdc_id
        )
        candidate_rows.append(row)
        missing = [name for name in NUTRIENTS if row[name] == ""]
        if missing:
            print(f"WARNING FDC {fdc_id}: missing {', '.join(missing)}")
        else:
            print(f"Fetched {source['food']} (FDC {fdc_id})")

    candidate_path = args.output / "candidate_foods.csv"
    with candidate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    print(f"Candidate table: {candidate_path}")
    print("Review descriptions, serving scaling and nutrients before using it.")


if __name__ == "__main__":
    main()
