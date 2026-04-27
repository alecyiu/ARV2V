"""Single-clip V2V reconstruction with Wan 2.1 VACE 1.3B.

Loads one Waymo clip, feeds it through VACE as the control video, writes the
reconstructed mp4 plus a side-by-side comparison and a meta.json into a fresh
run folder under outputs/v2v_recon/.

Usage:
    python inference/v2v_recon.py --video-id 044
    python inference/v2v_recon.py --video-id waymo_000044 --steps 30 --seed 0
    python inference/v2v_recon.py --video-id 044 --no-side-by-side
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

from config import (
    DEFAULT_FPS_OUT,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_PROMPT,
    DEFAULT_WIDTH,
    V2V_RECON_OUTPUTS,
    WAN_VACE_1_3B_LOCAL,
    video_path_for,
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_video_as_pil(path: Path, num_frames: int) -> tuple[list[Image.Image], float]:
    """Read up to `num_frames` frames as PIL.Image.RGB. Return frames + source fps."""
    meta = iio.immeta(path, plugin="pyav")
    src_fps = float(meta.get("fps", DEFAULT_FPS_OUT))
    frames: list[Image.Image] = []
    for frame in iio.imiter(path, plugin="pyav"):
        # imiter returns HxWx3 uint8 RGB
        frames.append(Image.fromarray(frame))
        if len(frames) == num_frames:
            break
    if len(frames) < num_frames:
        raise ValueError(
            f"{path.name}: only {len(frames)} frames available, need {num_frames}"
        )
    return frames, src_fps


def resize_letterbox(frames: list[Image.Image], target_h: int, target_w: int) -> list[Image.Image]:
    """Center-crop + resize each frame to target HxW preserving aspect ratio."""
    out: list[Image.Image] = []
    target_ratio = target_w / target_h
    for img in frames:
        w, h = img.size
        src_ratio = w / h
        # First crop to target aspect, then resize
        if src_ratio > target_ratio:
            # too wide -> crop width
            new_w = int(round(h * target_ratio))
            x0 = (w - new_w) // 2
            img = img.crop((x0, 0, x0 + new_w, h))
        elif src_ratio < target_ratio:
            # too tall -> crop height
            new_h = int(round(w / target_ratio))
            y0 = (h - new_h) // 2
            img = img.crop((0, y0, w, y0 + new_h))
        out.append(img.resize((target_w, target_h), Image.BICUBIC))
    return out


def export_video(frames: list[Image.Image], path: Path, fps: float) -> None:
    arr = np.stack([np.asarray(f) for f in frames], axis=0)  # (T, H, W, 3) uint8
    iio.imwrite(path, arr, plugin="pyav", fps=fps, codec="h264")


def export_side_by_side(
    left: list[Image.Image], right: list[Image.Image], path: Path, fps: float
) -> None:
    pairs = [
        Image.fromarray(np.concatenate([np.asarray(l), np.asarray(r)], axis=1))
        for l, r in zip(left, right)
    ]
    export_video(pairs, path, fps)


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------
def load_pipeline(model_path: Path, dtype: torch.dtype):
    """Import and load the VACE pipeline. Imports are local so non-inference
    callers (e.g. the unit-test or a list-runs helper) don't pull in diffusers.
    """
    try:
        from diffusers import WanVACEPipeline  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "diffusers WanVACEPipeline not found — install/upgrade with:\n"
            "  uv pip install -U 'diffusers>=0.34' transformers accelerate "
            "huggingface_hub safetensors"
        ) from e

    pipe = WanVACEPipeline.from_pretrained(str(model_path), torch_dtype=dtype)
    # Sequential CPU offload keeps peak VRAM low (works fine for 1.3B on a
    # single A6000 with ~20 GB free; remove for max speed if 30+ GB is free).
    pipe.enable_model_cpu_offload()
    return pipe


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video-id", required=True,
                   help="Short id ('44', '000044', 'waymo_000044') or full filename.")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    p.add_argument("--steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS)
    p.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE_SCALE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-path", default=str(WAN_VACE_1_3B_LOCAL),
                   help="Local path to the downloaded VACE 1.3B snapshot.")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--no-side-by-side", action="store_true",
                   help="Skip writing side_by_side.mp4.")
    p.add_argument("--out-fps", type=float, default=None,
                   help="Output fps (default: source fps, falling back to 10).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    src_path = video_path_for(args.video_id)
    if not src_path.exists():
        print(f"ERROR: source video not found: {src_path}", file=sys.stderr)
        return 1

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}. "
              f"Run `python inference/download_model.py` first.", file=sys.stderr)
        return 1

    # Per-run output directory
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = V2V_RECON_OUTPUTS / f"{stamp}_{src_path.stem}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Symlink the input so the run folder is self-contained
    input_link = run_dir / "input.mp4"
    if input_link.exists() or input_link.is_symlink():
        input_link.unlink()
    input_link.symlink_to(src_path)

    print(f"[1/4] Loading source: {src_path}")
    src_frames, src_fps = load_video_as_pil(src_path, args.num_frames)
    out_fps = args.out_fps if args.out_fps is not None else src_fps
    print(f"      {len(src_frames)} frames @ {src_fps} fps "
          f"(model expects {args.num_frames}); output fps = {out_fps}")

    print(f"[2/4] Resizing to {args.width}x{args.height}")
    cond_frames = resize_letterbox(src_frames, args.height, args.width)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    print(f"[3/4] Loading pipeline ({model_path}, {args.dtype})")
    t0 = time.time()
    pipe = load_pipeline(model_path, dtype)
    load_time = time.time() - t0
    print(f"      pipeline ready in {load_time:.1f}s")

    print(f"[4/4] Generating ({args.num_frames} frames, {args.steps} steps, "
          f"guidance={args.guidance}, seed={args.seed})")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    t0 = time.time()
    result = pipe(
        prompt=args.prompt,
        video=cond_frames,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    )
    gen_time = time.time() - t0
    out_frames = result.frames[0]  # diffusers convention: List[List[PIL.Image]]
    print(f"      generated in {gen_time:.1f}s ({gen_time/args.num_frames:.2f}s/frame)")

    # Write outputs
    output_path = run_dir / "output.mp4"
    export_video(out_frames, output_path, fps=out_fps)
    print(f"      wrote {output_path}")

    sbs_path: Path | None = None
    if not args.no_side_by_side:
        # cond_frames is already at the model's resolution, matches output dims
        sbs_path = run_dir / "side_by_side.mp4"
        export_side_by_side(cond_frames, list(out_frames), sbs_path, fps=out_fps)
        print(f"      wrote {sbs_path}")

    meta = {
        "input": str(src_path),
        "input_fps": src_fps,
        "output_fps": out_fps,
        "model_path": str(model_path),
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "steps": args.steps,
        "guidance_scale": args.guidance,
        "seed": args.seed,
        "dtype": args.dtype,
        "load_time_s": round(load_time, 2),
        "gen_time_s": round(gen_time, 2),
        "timestamp_utc": stamp,
        "outputs": {
            "input_link": str(input_link),
            "output": str(output_path),
            "side_by_side": str(sbs_path) if sbs_path else None,
        },
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone. Run folder: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
