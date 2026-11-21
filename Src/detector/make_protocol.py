"""Create protocol.json for a trained detector checkpoint (hash, class order, thresholds).

    python -m Src.detector.make_protocol --checkpoint runs/e6/NAME/weights/best.pt \\
        --out Models/e6/protocol.json --conf 0.25 --iou 0.6 --run-manifest runs/e6/NAME/run_manifest.json

Pick --conf using the VALIDATION split only. The protocol records scores as uncalibrated.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .classes import CLASS_NAMES
from .e6_detector import PROTOCOL_VERSION
from .yolo_io import sha256_file


def build_protocol(
    checkpoint: Path,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    run_manifest: Path | None = None,
) -> dict[str, Any]:
    protocol: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "class_names": list(CLASS_NAMES),
        "imgsz": imgsz,
        "conf_threshold": conf,
        "iou_threshold": iou,
        "max_det": max_det,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_file": checkpoint.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "calibrated": False,
        "notes": "Raw detector scores. No calibration and no unknown-object rejection have been validated.",
    }
    if run_manifest is not None:
        protocol["run_manifest_sha256"] = sha256_file(run_manifest)
        try:
            manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
            protocol["training_versions"] = manifest.get("versions")
            protocol["dataset_data_yaml_sha256"] = manifest.get("dataset_check", {}).get("data_yaml_sha256")
        except (OSError, ValueError):
            pass
    return protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a detector protocol.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--run-manifest")
    args = parser.parse_args(argv)

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"Checkpoint not found: {ckpt}")
        return 1
    protocol = build_protocol(
        ckpt,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        run_manifest=Path(args.run_manifest) if args.run_manifest else None,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Wrote {out}  (checkpoint sha256 {protocol['checkpoint_sha256']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
