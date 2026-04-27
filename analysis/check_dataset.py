"""Cross-check Waymo train metadata against the videos actually present on disk.

Runs as a script (`python check_dataset.py`) and is also importable from the
companion notebook (`check_dataset.ipynb`).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

WAYMO_ROOT = Path("/miele/liory/datasets/waymo")
META = WAYMO_ROOT / "train" / "metadata_vace_general.csv"
VIDEOS_DIR = WAYMO_ROOT / "videos"

# Per the dataset's metadata convention, camera_id=0 is the forward camera.
# Positions of the other camera_ids are not documented in the source notes.
CAMERA_LABELS = {"0": "front"}


@dataclass
class Stats:
    total_rows: int = 0
    files_on_disk: int = 0
    per_cam_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_cam_present: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_cam_scenes: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))


def present_videos(videos_dir: Path = VIDEOS_DIR) -> set[str]:
    return {p.name for p in videos_dir.iterdir() if p.suffix == ".mp4"}


def analyze(meta_path: Path = META, present: set[str] | None = None) -> Stats:
    if present is None:
        present = present_videos()

    stats = Stats(files_on_disk=len(present))
    with meta_path.open(newline="") as f:
        for row in csv.DictReader(f):
            stats.total_rows += 1
            cam = row["camera_id"]
            stats.per_cam_total[cam] += 1
            fname = row["video"].rsplit("/", 1)[-1]
            if fname in present:
                stats.per_cam_present[cam] += 1
                stats.per_cam_scenes[cam].add(row["scene_id"])
    return stats


def format_summary(stats: Stats) -> str:
    lines = [
        f"Train metadata rows: {stats.total_rows}",
        f"Videos on disk:      {stats.files_on_disk}",
        "",
        f"{'cam_id':<14}  {'meta_rows':>10}  {'with_file':>10}  {'coverage':>9}  {'scenes':>7}",
        f"{'-'*14:<14}  {'-'*10:>10}  {'-'*10:>10}  {'-'*9:>9}  {'-'*7:>7}",
    ]
    for cam in sorted(stats.per_cam_total):
        meta = stats.per_cam_total[cam]
        have = stats.per_cam_present.get(cam, 0)
        pct = 100 * have / meta if meta else 0
        scenes = len(stats.per_cam_scenes.get(cam, set()))
        label = f"{cam} ({CAMERA_LABELS[cam]})" if cam in CAMERA_LABELS else cam
        lines.append(f"{label:<14}  {meta:>10}  {have:>10}  {pct:>8.1f}%  {scenes:>7}")

    front_have = stats.per_cam_present.get("0", 0)
    front_total = stats.per_cam_total.get("0", 0)
    lines += [
        "",
        f"Front camera (camera_id=0): {front_have} of {front_total} metadata rows have a video file on disk",
    ]
    return "\n".join(lines)


def main() -> None:
    print(format_summary(analyze()))


if __name__ == "__main__":
    main()
