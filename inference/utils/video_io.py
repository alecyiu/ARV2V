"""Frame-level video I/O: read, center-crop/resize, dtype coercion, write."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

from config import DEFAULT_FPS_OUT


def read_video(path: Path, num_frames: int) -> tuple[list[Image.Image], float]:
    """Read up to `num_frames` RGB uint8 PIL frames; return (frames, source_fps)."""
    src_fps = float(iio.immeta(path, plugin="pyav").get("fps", DEFAULT_FPS_OUT))
    frames: list[Image.Image] = []
    for frame in iio.imiter(path, plugin="pyav"):
        frames.append(Image.fromarray(frame))
        if len(frames) == num_frames:
            break
    if len(frames) < num_frames:
        raise ValueError(f"{path.name}: only {len(frames)} frames, need {num_frames}")
    return frames, src_fps


def center_crop_resize(frames: list[Image.Image], h: int, w: int) -> list[Image.Image]:
    """Center-crop each frame to the target aspect, then bicubic-resize to (w, h)."""
    target = w / h
    out = []
    for img in frames:
        iw, ih = img.size
        ratio = iw / ih
        if ratio > target:
            nw = int(round(ih * target))
            img = img.crop(((iw - nw) // 2, 0, (iw - nw) // 2 + nw, ih))
        elif ratio < target:
            nh = int(round(iw / target))
            img = img.crop((0, (ih - nh) // 2, iw, (ih - nh) // 2 + nh))
        out.append(img.resize((w, h), Image.BICUBIC))
    return out


def to_pil_uint8(frame) -> Image.Image:
    """Coerce a PIL/numpy frame of any dtype to RGB uint8 PIL."""
    if isinstance(frame, Image.Image):
        return frame if frame.mode == "RGB" else frame.convert("RGB")
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr)


def write_video(frames: list[Image.Image], path: Path, fps: float) -> None:
    """Encode a list of same-size RGB PIL frames to an h264 mp4 at the given fps."""
    arr = np.stack([np.asarray(f) for f in frames], axis=0)
    iio.imwrite(path, arr, plugin="pyav", fps=fps, codec="h264")


def write_side_by_side(left, right, path: Path, fps: float) -> None:
    """Concat two same-length frame lists horizontally and write a single mp4."""
    pairs = [
        Image.fromarray(np.concatenate([np.asarray(l), np.asarray(r)], axis=1))
        for l, r in zip(left, right)
    ]
    write_video(pairs, path, fps)
