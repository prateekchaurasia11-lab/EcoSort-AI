"""Train the E6 YOLO detector after a mandatory dataset check, and record provenance.

    python -m Src.detector.train_yolo --data /content/det_data/data.yaml --epochs 100 --name e6_yolov8n_direct6

Writes <run dir>/run_manifest.json with arguments, dataset summary, data.yaml hash,
library versions and the best checkpoint's SHA-256. Choose settings on the validation
split only. Never tune anything on the test split.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .check_dataset import check_dataset
from .yolo_io import sha256_file


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for module in ("ultralytics", "torch", "torchvision", "numpy", "PIL"):
        try:
            mod = __import__(module)
            out[module] = str(getattr(mod, "__version__", "unknown"))
        except ImportError:
            out[module] = "not installed"
    try:
        import torch

        out["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu only"
    except Exception:  # noqa: BLE001 - versions are best-effort provenance
        out["cuda_device"] = "unknown"
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the E6 detector (direct six-class YOLO).")
    parser.add_argument("--data", required=True)
    parser.add_argument("--manifest", help="optional group manifest CSV (file_name, group_id) for the leakage check")
    parser.add_argument("--model", default="yolov8n.pt", help="starting weights or model yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu; default lets Ultralytics choose")
    parser.add_argument("--project", default="runs/e6")
    parser.add_argument("--name", default="e6_yolov8n_direct6")
    args = parser.parse_args(argv)

    report = check_dataset(args.data, manifest_csv=args.manifest, imgsz=args.imgsz)
    if not report["ok"]:
        print("Dataset check FAILED. Fix the dataset before training:")
        for message in report["errors"][:25]:
            print(f"  ERROR: {message}")
        return 1
    print("Dataset check passed.")
    for warning in report["warnings"][:10]:
        print(f"  warning: {warning}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("The 'ultralytics' package is not installed. Run: python -m pip install ultralytics")
        return 2

    started = datetime.now(timezone.utc)
    model = YOLO(args.model)
    train_kwargs: dict[str, Any] = {
        "data": str(Path(args.data).resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "deterministic": True,
        "patience": args.patience,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "exist_ok": False,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    model.train(**train_kwargs)

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    manifest: dict[str, Any] = {
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "train_kwargs": train_kwargs,
        "dataset_check": {
            "data_yaml_sha256": report.get("data_yaml_sha256"),
            "splits": report["splits"],
            "class_boxes_total": report["class_boxes_total"],
            "warnings": report["warnings"],
        },
        "versions": _versions(),
        "save_dir": str(save_dir),
        "note": "Select checkpoints and thresholds on val only. Report test once, unchanged.",
    }
    if best.is_file():
        manifest["best_pt"] = {"path": str(best), "sha256": sha256_file(best), "bytes": best.stat().st_size}
    else:
        manifest["best_pt"] = None
        print(f"WARNING: {best} not found; training may have failed.")
    (save_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Run manifest written to {save_dir / 'run_manifest.json'}")
    return 0 if best.is_file() else 3


if __name__ == "__main__":
    sys.exit(main())
