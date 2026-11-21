"""Concurrent downloader for official TACO images referenced in annotations.json."""
from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _download_one(img: dict, out_root: Path, use_640: bool, timeout: float) -> tuple[str, str]:
    rel_path = out_root / img["file_name"]
    if rel_path.is_file() and rel_path.stat().st_size > 0:
        return "cached", img["file_name"]

    url = (img.get("flickr_640_url") if use_640 else None) or img.get("flickr_url")
    if not url:
        return "no_url", img["file_name"]

    rel_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = rel_path.with_suffix(rel_path.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "EcoSort-AI-DatasetPrep/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp_path.open("wb") as out_f:
            out_f.write(resp.read())
        tmp_path.replace(rel_path)
        return "downloaded", img["file_name"]
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return "failed", img["file_name"]


def download_taco(
    coco_json: Path,
    out_root: Path,
    limit: int | None = None,
    workers: int = 8,
    use_640: bool = False,
    timeout: float = 20.0,
) -> int:
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    images = data.get("images", [])
    if limit is not None and limit > 0:
        images = images[:limit]

    counts = {"downloaded": 0, "cached": 0, "failed": 0, "no_url": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_download_one, img, out_root, use_640, timeout) for img in images]
        for idx, fut in enumerate(as_completed(futures), start=1):
            status, _ = fut.result()
            counts[status] += 1
            if idx % 100 == 0 or idx == len(images):
                print(f"Progress {idx}/{len(images)}: {counts}")

    print(f"Finished TACO fetch into {out_root}: {counts}")
    return 0 if counts["failed"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Download TACO images into Data/taco_raw.")
    parser.add_argument("--coco-json", type=Path, default=Path("Data/taco_raw/annotations.json"))
    parser.add_argument("--out", type=Path, default=Path("Data/taco_raw"))
    parser.add_argument("--limit", type=int, default=None, help="Optional limit (e.g. 50 for smoke test)")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent download workers")
    parser.add_argument("--use-640", action="store_true", help="Download 640px thumbnails instead of full-res")
    args = parser.parse_args()
    return download_taco(args.coco_json, args.out, args.limit, args.workers, args.use_640)


if __name__ == "__main__":
    raise SystemExit(main())
