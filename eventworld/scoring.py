"""CPU scoring for the released EventScape protocol."""
from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
from statistics import fmean

import numpy as np

from .assets import HEIGHT, WIDTH, asset, read_json, sha256, write_json


def score_depth(depth: np.ndarray, truth: np.ndarray, semantic: np.ndarray) -> dict:
    prediction = np.asarray(depth, dtype=np.float64)
    reference = np.asarray(truth, dtype=np.float64)
    if prediction.shape != (HEIGHT, WIDTH) or reference.shape != prediction.shape or semantic.shape != prediction.shape:
        raise ValueError("Depth and semantic labels must share the native 252x504 crop")
    if not np.isfinite(prediction).all() or np.any(prediction <= 0):
        raise ValueError("Depth values must be finite and positive")
    mask = (semantic != 0) & np.isfinite(reference) & (reference > 0) & (reference <= 30)
    pixels = int(mask.sum())
    if pixels < 100:
        raise ValueError("At least 100 valid non-sky pixels within 30 m are required")
    values, targets = prediction[mask], reference[mask]
    index = (pixels - 1) // 2
    scale = float(np.partition(targets, index)[index] / np.partition(values, index)[index])
    values *= scale
    return {
        "pixels": pixels,
        "scale": scale,
        "absrel": float(np.mean(np.abs(values - targets) / targets)),
        "delta1": float(np.mean(np.maximum(values / targets, targets / values) < 1.25)),
    }


def aggregate(rows: list[dict]) -> list[dict]:
    by_recording = defaultdict(list)
    for row in rows:
        by_recording[row["horizon_ms"], row["recording_group"]].append(row)
    by_horizon = defaultdict(list)
    for (horizon, _), group in sorted(by_recording.items()):
        by_horizon[horizon].append({key: fmean(row[key] for row in group) for key in ("absrel", "delta1")})
    return [
        {"horizon_ms": horizon, "recordings": len(group),
         **{key: fmean(row[key] for row in group) for key in ("absrel", "delta1")}}
        for horizon, group in sorted(by_horizon.items())
    ]


def score_predictions(dataset: Path, predictions: Path, output: Path) -> dict:
    generation = read_json(predictions / "generation.json")
    if generation.get("format_version") != 1 or not generation["predictions"]:
        raise ValueError("Expected a complete prediction manifest")
    targets = read_json(dataset / "targets/index.json")
    if targets.get("format_version") != 1:
        raise ValueError("Unsupported EventScape target index")
    targets_by_id = {row["case_id"]: row for row in targets["cases"]}
    horizons = generation["horizons_ms"]
    if not horizons or len(set(horizons)) != len(horizons) or not set(horizons) <= set(targets["horizons_ms"]):
        raise ValueError("The requested horizons must be present in the released targets")
    case_ids = [item["case_id"] for item in generation["predictions"]]
    if len(case_ids) != len(set(case_ids)) or not set(case_ids) <= targets_by_id.keys():
        raise ValueError("Duplicate or unknown prediction case")
    rows = []
    for item in generation["predictions"]:
        identity = item["case_id"]
        case = targets_by_id[identity]
        path = asset(predictions, item["file"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Prediction checksum mismatch: {identity}")
        with np.load(path, allow_pickle=False) as data:
            depth = data["depth"]
            if data["horizons_ms"].tolist() != horizons or depth.shape != (len(horizons), HEIGHT, WIDTH):
                raise ValueError(f"Unexpected prediction axes for {identity}")
        for index, horizon in enumerate(horizons):
            target = case["targets"][str(horizon)]
            if abs(target["matched_us"] - target["query_us"]) > 2:
                raise ValueError(f"Unexpected target timestamp: {identity} at {horizon} ms")
            truth = np.load(asset(dataset, target["depth"]), allow_pickle=False)
            semantic = np.load(asset(dataset, target["semantic"]), allow_pickle=False)
            metrics = score_depth(depth[index], truth, semantic)
            rows.append({"case_id": identity, "recording_group": case["recording_group"],
                         "horizon_ms": horizon, **metrics})

    statistics = aggregate(rows)
    report = {
        "cases": len(case_ids),
        "targets": len(rows),
        "all_dataset_cases": set(case_ids) == set(targets_by_id),
        "recording_macro": statistics,
        "mean_over_horizons": {key: fmean(row[key] for row in statistics) for key in ("absrel", "delta1")},
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "per_case.csv").open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "summary.json", report)
    return report
