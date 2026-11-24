"""Command line entry points for inference and reproducible evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .assets import (
    DEFAULT_HORIZONS, EVENTSCAPE_HORIZONS, asset, load_case_images, load_motion, read_cases, sha256,
    verify_assets, write_json,
)


def model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weights", type=Path, required=True, help="EventWorld-P40530.safetensors")
    parser.add_argument("--eventvggt", type=Path, required=True, help="Upstream EventVGGT checkpoint.pt")
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS, metavar="MS")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, prog="python -m eventworld")
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser("infer", help="Forecast one window from prepared past events")
    model_options(infer)
    source = infer.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="Extracted EventScape test data")
    source.add_argument("--images", type=Path, nargs=3, metavar="PNG", help="Three event images, oldest first")
    infer.add_argument("--case", help="Case ID in --data, defaults to its first case")
    infer.add_argument("--features", type=Path, help="TESPEC feature NPZ paired with --images")
    infer.add_argument("--seed", type=int, help="Override the dataset seed or the custom-input default")
    infer.add_argument("--camera-checkpoint", type=Path, help="Upstream EventVGGT checkpoint_pose.pt")
    infer.add_argument("--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"),
                       help="Also return a camera-frame point cloud from forecast depth")

    test = commands.add_parser("test", help="Generate and score EventScape depth forecasts")
    model_options(test)
    test.add_argument("--data", type=Path, required=True)
    test.add_argument("--limit", type=int, help="Run a quick check on the first N cases")

    score = commands.add_parser("score", help="Score a completed test prediction bundle on CPU")
    score.add_argument("--data", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)

    check = commands.add_parser("verify-data", help="Check every SHA-256 in the extracted EventScape archive")
    check.add_argument("--data", type=Path, required=True)
    return parser.parse_args()


def write_prediction(directory: Path, identity: str, result: dict) -> dict:
    name = f"{identity}.npz"
    path = directory / name
    with path.open("xb") as file:
        np.savez_compressed(file, **result)
    return {"case_id": identity, "file": name, "sha256": sha256(path)}


def run_inference(args: argparse.Namespace, horizons: tuple[int, ...]) -> None:
    from .runtime import EventWorld, backproject

    if args.data:
        if args.features:
            raise ValueError("--features is only needed with --images")
        cases = read_cases(args.data)
        identity = args.case or next(iter(cases))
        row = cases[identity]
        paths = [asset(args.data, name) for name in row["events"]]
        motion = asset(args.data, row["ordered_features"])
        default_seed = row["seed"]
    else:
        if args.features is None or args.case is not None:
            raise ValueError("Use --images with --features; --case selects an item from --data")
        identity, paths, motion, default_seed = "forecast", args.images, args.features, 42

    images = load_case_images(paths)
    features = load_motion(motion)
    seed = default_seed if args.seed is None else args.seed
    model = EventWorld(args.weights, args.eventvggt, args.camera_checkpoint)
    result = model.forecast(images, features, horizons, seed)
    if args.intrinsics is not None:
        if not np.isfinite(args.intrinsics).all() or min(args.intrinsics[:2]) <= 0:
            raise ValueError("Expected finite intrinsics with positive focal lengths")
        result["xyz_camera"] = backproject(result["depth"], tuple(args.intrinsics))
    args.output.mkdir(parents=True, exist_ok=False)
    info = write_prediction(args.output, identity, result)
    print(json.dumps({"prediction": str(args.output / info["file"]), "horizons_ms": horizons,
                      "depth_shape": list(result["depth"].shape), "seed": seed}, indent=2))


def run_test(args: argparse.Namespace, horizons: tuple[int, ...]) -> None:
    from .runtime import EventWorld
    from .scoring import score_predictions

    cases = list(read_cases(args.data).values())
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        cases = cases[:args.limit]
    model = EventWorld(args.weights, args.eventvggt)
    predictions = args.output / "predictions"
    args.output.mkdir(parents=True, exist_ok=False)
    predictions.mkdir()
    generated = []
    for index, row in enumerate(cases, start=1):
        images = load_case_images([asset(args.data, name) for name in row["events"]])
        motion = load_motion(asset(args.data, row["ordered_features"]))
        result = model.forecast(images, motion, horizons, row["seed"])
        generated.append(write_prediction(predictions, row["case_id"], result))
        print(f"{index}/{len(cases)} {row['case_id']}", flush=True)

    # Ground truth is first accessed by the scorer, after all forecasts are sealed.
    write_json(predictions / "generation.json", {
        "format_version": 1, "horizons_ms": list(horizons), "predictions": generated,
    })
    print(json.dumps(score_predictions(args.data, predictions, args.output / "scores"), indent=2))


def main() -> None:
    args = options()
    if args.command == "verify-data":
        print(f"Verified {verify_assets(args.data)} dataset files")
        return
    if args.command == "score":
        from .scoring import score_predictions
        print(json.dumps(score_predictions(args.data, args.predictions, args.output), indent=2))
        return
    horizons = tuple(args.horizons)
    if not horizons or len(set(horizons)) != len(horizons) or any(h < 100 or h > 800 for h in horizons):
        raise ValueError("Use distinct integer forecast horizons between 100 and 800 ms")
    if args.command == "infer":
        run_inference(args, horizons)
    else:
        if not set(horizons) <= set(EVENTSCAPE_HORIZONS):
            raise ValueError("EventScape test targets are available from 120 to 800 ms at 40 ms intervals")
        run_test(args, horizons)


if __name__ == "__main__":
    main()
