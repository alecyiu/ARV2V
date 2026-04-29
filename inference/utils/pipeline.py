"""WanVACE pipeline load + a single-clip inference wrapper."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from PIL import Image

from .video_io import to_pil_uint8

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def load_pipeline(model_path: Path, dtype: torch.dtype, gpu_id: int):
    """Load `WanVACEPipeline` with `enable_model_cpu_offload` pinned to `gpu_id`."""
    from diffusers import WanVACEPipeline  # type: ignore[import-not-found]

    pipe = WanVACEPipeline.from_pretrained(str(model_path), torch_dtype=dtype)
    pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    return pipe


def infer_one(
    pipe,
    *,
    prompt: str,
    negative_prompt: str,
    cond_frames: list[Image.Image],
    height: int,
    width: int,
    num_frames: int,
    steps: int,
    guidance: float,
    conditioning_scale: float,
    seed: int,
    gpu_id: int,
) -> tuple[list[Image.Image], float]:
    """Run one VACE inference; return (output PIL frames, gen_time_s)."""
    t0 = time.perf_counter()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        video=cond_frames,
        conditioning_scale=conditioning_scale,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=torch.Generator(device=f"cuda:{gpu_id}").manual_seed(seed),
    )
    gen_time = time.perf_counter() - t0
    out_frames = [to_pil_uint8(f) for f in result.frames[0]]
    return out_frames, gen_time
