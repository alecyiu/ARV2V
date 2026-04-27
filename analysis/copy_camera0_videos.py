"""Copy camera_id==0 videos + metadata from the canonical Waymo dataset on
`/miele/` (NFS) into a local mirror at `/home/alec/ARV2V/waymo/`, for both
train and val. Mirrors the source directory structure exactly.

Flow per split:
    1. Copy the metadata CSV to the destination (same relative path).
    2. Read the CSV, filter rows where `camera_id == 0`.
    3. Copy each referenced video from src to dst in parallel
       (NFS reads are latency-bound, so concurrency helps a lot).

Run:
    uv run analysis/copy_camera0_videos.py --dry-run
    uv run analysis/copy_camera0_videos.py
    uv run analysis/copy_camera0_videos.py --workers 32
"""

from __future__ import annotations

import argparse
import shutil
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

SRC_ROOT = Path("/miele/liory/datasets/waymo")
DST_ROOT = Path("/home/alec/ARV2V/waymo")

SPLITS = {
    "train": {"meta": "train/metadata_vace_general.csv",            "vbase": ""},
    "val":   {"meta": "val/metadata_vace_general_stride_global.csv", "vbase": "val"},
}


def copy_one(src: Path, dst: Path, verify: bool) -> tuple[str, int]:
    """Returns (status, bytes_copied). status in {'copied','skipped','missing'}.

    Fast resume: if dst exists locally, skip without touching the NFS source
    (one NFS stat per file adds up). Pass `verify=True` to also size-check
    against the source (slower, but catches partial copies).
    Writes go to `<dst>.tmp` then atomic rename so future runs never see a
    half-written file.
    """
    if dst.exists():
        if not verify:
            return "skipped", 0
        if dst.stat().st_size == src.stat().st_size:
            return "skipped", 0
        # size mismatch -> fall through and re-copy

    if not src.exists():
        return "missing", 0

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.rename(dst)
    return "copied", dst.stat().st_size


def copy_metadata(split: str, dry_run: bool) -> None:
    cfg = SPLITS[split]
    src = SRC_ROOT / cfg["meta"]
    dst = DST_ROOT / cfg["meta"]
    print(f"[{split}] metadata: {src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_videos(split: str, dry_run: bool, workers: int, verify: bool) -> None:
    cfg = SPLITS[split]
    src_meta = SRC_ROOT / cfg["meta"]
    src_vbase = SRC_ROOT / cfg["vbase"]
    dst_vbase = DST_ROOT / cfg["vbase"]

    df = pd.read_csv(src_meta, usecols=["video", "camera_id"])
    rel_paths = df.loc[df["camera_id"] == 0, "video"].tolist()
    print(f"[{split}] camera_id==0 videos: {len(rel_paths):,}")
    print(f"[{split}] src: {src_vbase}   dst: {dst_vbase}   workers: {workers}")

    if dry_run:
        return

    pairs = [(src_vbase / r, dst_vbase / r) for r in rel_paths]

    copied = skipped = missing = 0
    bytes_copied = 0
    t0 = time.time()
    # Rolling window of (timestamp, files_done, bytes_done) for instantaneous rate.
    window: deque[tuple[float, int, int]] = deque(maxlen=400)
    window.append((t0, 0, 0))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(copy_one, s, d, verify) for s, d in pairs]
        for i, fut in enumerate(as_completed(futures), 1):
            status, nbytes = fut.result()
            if status == "copied":
                copied += 1
                bytes_copied += nbytes
            elif status == "skipped":
                skipped += 1
            else:
                missing += 1

            if i % 200 == 0 or i == len(futures):
                now = time.time()
                window.append((now, i, bytes_copied))
                t_old, i_old, b_old = window[0]
                dt = max(now - t_old, 1e-6)
                inst_rate = (i - i_old) / dt
                inst_mbps = (bytes_copied - b_old) / dt / 1e6
                cum_rate = i / max(now - t0, 1e-6)
                print(
                    f"  [{split}] {i:,}/{len(futures):,}"
                    f"  copied={copied:,} skipped={skipped:,} missing={missing:,}"
                    f"  inst={inst_rate:.1f} f/s, {inst_mbps:.1f} MB/s"
                    f"  (cum {cum_rate:.1f} f/s)"
                )

    total_min = (time.time() - t0) / 60
    print(
        f"[{split}] done in {total_min:.1f} min:"
        f"  copied={copied:,}  skipped={skipped:,}  missing={missing:,}"
        f"  total={bytes_copied / 1e9:.2f} GB"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel copy workers. NFS reads are latency-bound, so 16-32 helps a lot.",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Size-check existing dst files against src (slower; useful once after "
        "an interrupted run to catch partial files).",
    )
    args = ap.parse_args()

    print(f"src root: {SRC_ROOT}")
    print(f"dst root: {DST_ROOT}")
    if args.dry_run:
        print("(dry run — no files will be written)")

    for split in args.splits:
        print(f"\n=== {split} ===")
        copy_metadata(split, dry_run=args.dry_run)
        copy_videos(split, dry_run=args.dry_run, workers=args.workers, verify=args.verify)


if __name__ == "__main__":
    main()
