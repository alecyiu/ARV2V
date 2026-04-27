"""Single-clip V2V reconstruction with Wan 2.1 VACE 1.3B.

Loads one Waymo clip, feeds it through VACE as the control video, writes the
reconstructed mp4 plus a side-by-side comparison and a meta.json into a fresh
run folder under outputs/v2v_recon/.

Usage:
    python inference/v2v_recon.py --video-id 0
    python inference/v2v_recon.py --video-id waymo_000044 --steps 30 --gpu 3
    python inference/v2v_recon.py --video-id 44 --no-side-by-side
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
    DEFAULT_CONDITIONING_SCALE,
    DEFAULT_FPS_OUT,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_PROMPT,
    DEFAULT_WIDTH,
    V2V_RECON_OUTPUTS,
    WAN_VACE_1_3B_LOCAL,
    video_path_for,
)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


# --- frame I/O ---------------------------------------------------------------

def read_video(path: Path, num_frames: int) -> tuple[list[Image.Image], float]:
    """Read up to `num_frames` RGB uint8 frames; return frames + source fps."""
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
    """Center-crop to target aspect, then resize to HxW."""
    target = w / h
    out = []
    for img in frames:
        iw, ih = img.size
        ratio = iw / ih
        if ratio > target:  # too wide -> crop width
            nw = int(round(ih * target))
            img = img.crop(((iw - nw) // 2, 0, (iw - nw) // 2 + nw, ih))
        elif ratio < target:  # too tall -> crop height
            nh = int(round(iw / target))
            img = img.crop((0, (ih - nh) // 2, iw, (ih - nh) // 2 + nh))
        out.append(img.resize((w, h), Image.BICUBIC))
    return out


def to_pil_uint8(frame) -> Image.Image:
    """Coerce a PIL/numpy frame (any dtype) to RGB uint8 PIL."""
    if isinstance(frame, Image.Image):
        return frame if frame.mode == "RGB" else frame.convert("RGB")
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:  # diffusers video pipelines often emit float32 in [0, 1]
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr)


def write_video(frames: list[Image.Image], path: Path, fps: float) -> None:
    arr = np.stack([np.asarray(f) for f in frames], axis=0)
    iio.imwrite(path, arr, plugin="pyav", fps=fps, codec="h264")


def write_side_by_side(left, right, path: Path, fps: float) -> None:
    pairs = [
        Image.fromarray(np.concatenate([np.asarray(l), np.asarray(r)], axis=1))
        for l, r in zip(left, right)
    ]
    write_video(pairs, path, fps)


# --- pipeline ----------------------------------------------------------------

def load_pipeline(model_path: Path, dtype: torch.dtype, gpu_id: int):
    from diffusers import WanVACEPipeline  # type: ignore[import-not-found]
    pipe = WanVACEPipeline.from_pretrained(str(model_path), torch_dtype=dtype)
    pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    return pipe


# --- main --------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video-id", required=True,
                   help="Short id ('44'), padded ('000044'), or 'waymo_000044'.")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT,
                   help="What to steer AWAY from via classifier-free guidance. "
                        "Wan's default fights warm/over-saturated color cast; pass '' to disable.")
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    p.add_argument("--steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS)
    p.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE_SCALE)
    p.add_argument("--conditioning-scale", type=float, default=DEFAULT_CONDITIONING_SCALE,
                   help="VACE control strength. 1.0 = source video dominates (reconstruction); "
                        "lower (e.g. 0.5) lets the prompt actually edit the content.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index.")
    p.add_argument("--model-path", default=str(WAN_VACE_1_3B_LOCAL))
    p.add_argument("--no-side-by-side", action="store_true")
    p.add_argument("--out-fps", type=float, default=None,
                   help="Output fps (default: source fps).")
    return p.parse_args()


def validate(args: argparse.Namespace, src_path: Path, model_path: Path) -> None:
    """Fail fast on missing inputs or invalid device. Exits via SystemExit on error."""
    for label, p in [("source video", src_path), ("model", model_path)]:
        if not p.exists():
            sys.exit(f"ERROR: {label} not found: {p}")
    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA not available.")
    n = torch.cuda.device_count()
    if not 0 <= args.gpu < n:
        sys.exit(f"ERROR: --gpu {args.gpu} out of range (have {n} devices).")


def main() -> int:
    args = parse_args()
    src_path = video_path_for(args.video_id)
    model_path = Path(args.model_path)
    validate(args, src_path, model_path)

    # Run folder. Symlink the input so it's self-contained.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = V2V_RECON_OUTPUTS / f"{stamp}_{src_path.stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    input_link = run_dir / "input.mp4"
    input_link.unlink(missing_ok=True)
    input_link.symlink_to(src_path)

    # Read + condition input.
    src_frames, src_fps = read_video(src_path, args.num_frames)
    out_fps = src_fps if args.out_fps is None else args.out_fps
    cond_frames = center_crop_resize(src_frames, args.height, args.width)
    print(f"input  : {src_path}  ({len(src_frames)} frames @ {src_fps} fps)")
    print(f"resize : -> {args.width}x{args.height}")

    # Load pipeline.
    print(f"model  : {model_path} ({args.dtype}) on "
          f"cuda:{args.gpu} ({torch.cuda.get_device_name(args.gpu)})")
    t0 = time.perf_counter()
    pipe = load_pipeline(model_path, DTYPES[args.dtype], gpu_id=args.gpu)
    load_time = time.perf_counter() - t0
    print(f"         ready in {load_time:.1f}s")

    # Generate.
    print(f"infer  : {args.num_frames} frames, {args.steps} steps, "
          f"guidance={args.guidance}, seed={args.seed}")
    t0 = time.perf_counter()
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        video=cond_frames,
        conditioning_scale=args.conditioning_scale,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=torch.Generator(device=f"cuda:{args.gpu}").manual_seed(args.seed),
    )
    gen_time = time.perf_counter() - t0
    out_frames = [to_pil_uint8(f) for f in result.frames[0]]
    print(f"         done in {gen_time:.1f}s ({gen_time / args.num_frames:.2f}s/frame)")

    # Write outputs + meta.
    output_path = run_dir / "output.mp4"
    write_video(out_frames, output_path, fps=out_fps)
    sbs_path: Path | None = None
    if not args.no_side_by_side:
        sbs_path = run_dir / "side_by_side.mp4"
        write_side_by_side(cond_frames, out_frames, sbs_path, fps=out_fps)

    meta = {
        "input": str(src_path),
        "input_fps": src_fps,
        "output_fps": out_fps,
        "model_path": str(model_path),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "conditioning_scale": args.conditioning_scale,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "steps": args.steps,
        "guidance_scale": args.guidance,
        "seed": args.seed,
        "dtype": args.dtype,
        "gpu": args.gpu,
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
    print(f"\ndone -> {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
