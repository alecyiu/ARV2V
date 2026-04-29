"""Per-cell mp4 cache for the grid demo.

A cell is uniquely keyed by everything that affects its pixels: the source clip
id, prompt + negative prompt, generation params, dtype, and model snapshot.
Re-running the grid notebook with unchanged params decodes the cached mp4
instead of paying for a 3-4 minute inference per cell.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v3 as iio
from PIL import Image

from config import OUTPUTS_ROOT

CACHE_DIR = OUTPUTS_ROOT / "v2v_recon" / "_cache" / "cells"
CACHE_VERSION = "v1"


def cache_key(
    *,
    video_id: str,
    preset_name: str,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_frames: int,
    steps: int,
    guidance: float,
    conditioning_scale: float,
    seed: int,
    dtype: str,
    model_path: Path,
) -> str:
    """SHA1 hex of all generation-affecting inputs; stable across runs and machines."""
    payload = {
        "version": CACHE_VERSION,
        "video_id": str(video_id),
        "preset_name": preset_name,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "steps": steps,
        "guidance": guidance,
        "conditioning_scale": conditioning_scale,
        "seed": seed,
        "dtype": dtype,
        "model_path": str(model_path),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(blob).hexdigest()


def cached_mp4_path(key: str) -> Path:
    """Return `CACHE_DIR / f"{key}.mp4"` (does not check existence)."""
    return CACHE_DIR / f"{key}.mp4"


def load_cached_frames(path: Path, num_frames: int) -> list[Image.Image] | None:
    """Decode `path` to PIL frames if it has at least `num_frames` decodable; else `None`."""
    if not path.exists():
        return None
    frames: list[Image.Image] = []
    try:
        for frame in iio.imiter(path, plugin="pyav"):
            frames.append(Image.fromarray(frame))
            if len(frames) == num_frames:
                break
    except Exception:
        return None
    if len(frames) < num_frames:
        return None
    return frames
