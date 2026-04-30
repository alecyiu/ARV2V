"""Grid demo runner — multi-GPU data-parallel cell dispatch.

`run_grid()` loads one Wan VACE pipeline per GPU in `gpu_ids` and fans cells
out across them via a thread pool. Total wall time scales roughly with 1/N
for N independent GPUs.

Per-cell mp4s are cached at outputs/v2v_recon/_cache/cells/<sha1>.mp4 keyed
by all generation params; the run dir gets a flat copy named
`<row_label>__<preset>.mp4` plus the composed `grid.mp4` and `meta.json`.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue

import torch

from config import V2V_RECON_OUTPUTS, WAN_VACE_1_3B_LOCAL, video_path_for
from prompts import PRESETS

from .cache import CACHE_DIR, cache_key, cached_mp4_path, load_cached_frames
from .grid import compose_grid_video
from .pipeline import DTYPES, infer_one, load_pipeline
from .video_io import center_crop_resize, read_video, write_video


def run_grid(
    *,
    video_ids: list[str],
    preset_names: list[str],
    gpu_ids: list[int],
    steps: int = 50,
    guidance: float = 7.5,
    conditioning_scale: float | dict[str, float] | None = None,
    seed: int = 0,
    height: int = 480,
    width: int = 832,
    num_frames: int = 49,
    dtype: str = "bf16",
    cell_w: int = 320,
    cell_h: int = 184,
    header_h: int = 32,
    label_w: int = 96,
    model_path: Path | str | None = None,
    save_per_cell: bool = True,
) -> dict:
    """Run the (rows x cols) grid demo across all listed GPUs.

    `conditioning_scale` resolution:
        - None  -> each preset uses its own `recommended_conditioning_scale`.
        - float -> uniform across all presets.
        - dict  -> per-preset override; falls back to recommended for missing keys.

    Returns a dict with `run_dir`, `grid_path`, `per_cell_paths`, and `meta`.
    """

    def cond_for(name: str) -> float:
        recommended = PRESETS[name].recommended_conditioning_scale
        if conditioning_scale is None:
            return recommended
        if isinstance(conditioning_scale, dict):
            return float(conditioning_scale.get(name, recommended))
        return float(conditioning_scale)
    model_path = Path(model_path or WAN_VACE_1_3B_LOCAL)

    # ---- validate -----------------------------------------------------------
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    n_devices = torch.cuda.device_count()
    if not gpu_ids:
        raise ValueError("gpu_ids must contain at least one device index.")
    for g in gpu_ids:
        if not 0 <= g < n_devices:
            raise ValueError(f"gpu id {g} out of range (have {n_devices} devices).")
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {list(DTYPES)}")
    for name in preset_names:
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}; choose from {list(PRESETS)}")

    video_paths = {vid: video_path_for(vid) for vid in video_ids}
    for vid, path in video_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"video {vid!r} not found: {path}")

    n_rows = len(video_ids)
    n_cols = 1 + len(preset_names)
    cond_by_preset = {name: cond_for(name) for name in preset_names}
    print(f"gpus  : {gpu_ids}  ({len(gpu_ids)} device{'s' if len(gpu_ids) > 1 else ''})")
    print(f"model : {model_path} ({dtype})")
    print(f"grid  : {n_rows} rows x {n_cols} cols "
          f"({n_rows * len(preset_names)} inferences)")
    print(f"params: guidance={guidance}  steps={steps}  seed={seed}")
    print(f"cond  : " + ", ".join(f"{n}={cond_by_preset[n]}" for n in preset_names))

    # ---- read GT clips once -------------------------------------------------
    gt_frames: dict[str, list] = {}
    out_fps: float | None = None
    print("clips :")
    for vid in video_ids:
        raw, fps = read_video(video_paths[vid], num_frames)
        gt_frames[vid] = center_crop_resize(raw, height, width)
        out_fps = fps
        print(f"  {vid:>3}: {len(raw)} frames @ {fps} fps")

    # ---- build work list + compute cache keys -------------------------------
    work = []
    for r, vid in enumerate(video_ids):
        for c_off, name in enumerate(preset_names):
            preset = PRESETS[name]
            cond = cond_by_preset[name]
            key = cache_key(
                video_id=vid, preset_name=name,
                prompt=preset.prompt, negative_prompt=preset.negative_prompt,
                height=height, width=width, num_frames=num_frames,
                steps=steps, guidance=guidance,
                conditioning_scale=cond,
                seed=seed, dtype=dtype, model_path=model_path,
            )
            work.append((r, 1 + c_off, vid, name, cond, key, cached_mp4_path(key)))

    n_hit = sum(1 for *_, p in work if p.exists())
    n_miss = len(work) - n_hit
    print(f"cache : {n_hit} hit / {n_miss} miss  ({CACHE_DIR})")

    # ---- fill cells from cache, collect misses ------------------------------
    cells: list[list[list | None]] = [[None] * n_cols for _ in range(n_rows)]
    for r, vid in enumerate(video_ids):
        cells[r][0] = gt_frames[vid]

    misses = []
    for entry in work:
        r, c, _vid, _name, _cond, _key, cache_path = entry
        cached = load_cached_frames(cache_path, num_frames)
        if cached is not None:
            cells[r][c] = cached
        else:
            misses.append(entry)

    # ---- lazy multi-GPU pipeline pool ---------------------------------------
    pipes_by_gpu: dict[int, object] = {}
    load_time = 0.0
    if misses:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"loading {len(gpu_ids)} pipeline"
              f"{'s' if len(gpu_ids) > 1 else ''}...")
        t_load = time.perf_counter()
        for gid in gpu_ids:
            pipe = load_pipeline(model_path, DTYPES[dtype], gpu_id=gid)
            try:
                pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass
            pipes_by_gpu[gid] = pipe
            print(f"  cuda:{gid} ready ({torch.cuda.get_device_name(gid)})")
        load_time = time.perf_counter() - t_load
        print(f"  total load {load_time:.1f}s")

    # ---- distribute misses across GPUs --------------------------------------
    gpu_q: Queue[int] = Queue()
    for gid in gpu_ids:
        gpu_q.put(gid)
    counter = {"done": 0}
    counter_lock = threading.Lock()

    def run_cell(entry):
        r, c, vid, name, cond, key, cache_path = entry
        gid = gpu_q.get()
        try:
            preset = PRESETS[name]
            t0 = time.perf_counter()
            out_frames, _ = infer_one(
                pipes_by_gpu[gid],
                prompt=preset.prompt,
                negative_prompt=preset.negative_prompt,
                cond_frames=gt_frames[vid],
                height=height, width=width, num_frames=num_frames,
                steps=steps, guidance=guidance,
                conditioning_scale=cond,
                seed=seed, gpu_id=gid,
            )
            elapsed = time.perf_counter() - t0
            write_video(out_frames, cache_path, fps=out_fps)
            with counter_lock:
                counter["done"] += 1
                idx = counter["done"]
            print(f"[{idx:>2}/{len(misses)}] {vid:>3} x {name:<10} "
                  f"cond={cond:<4} cuda:{gid}  {elapsed:.1f}s")
            return entry, out_frames
        finally:
            gpu_q.put(gid)

    run_t0 = time.perf_counter()
    if misses:
        print(f"inferring {len(misses)} cells across cuda:{gpu_ids}...")
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
            futures = {pool.submit(run_cell, e): e for e in misses}
            for f in as_completed(futures):
                entry, out_frames = f.result()
                r, c = entry[0], entry[1]
                cells[r][c] = out_frames
    run_time = time.perf_counter() - run_t0

    for r in range(n_rows):
        for c in range(n_cols):
            if cells[r][c] is None:
                raise RuntimeError(f"cell [{r}][{c}] not filled")

    # ---- run dir + grid + per-cell saves ------------------------------------
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = V2V_RECON_OUTPUTS / f"{stamp}_grid_demo"
    run_dir.mkdir(parents=True, exist_ok=True)

    col_labels = ["GT"] + [PRESETS[n].label for n in preset_names]
    row_labels = [f"waymo_{int(v):06d}" for v in video_ids]

    grid_path = run_dir / "grid.mp4"
    compose_grid_video(
        cells, grid_path, fps=out_fps,
        col_labels=col_labels, row_labels=row_labels,
        cell_w=cell_w, cell_h=cell_h, header_h=header_h, label_w=label_w,
    )

    per_cell_paths: dict[str, str] = {}
    input_links: dict[str, str] = {}
    if save_per_cell:
        for r, vid in enumerate(video_ids):
            clip_dir = run_dir / row_labels[r]
            clip_dir.mkdir(parents=True, exist_ok=True)
            input_link = clip_dir / "input.mp4"
            input_link.unlink(missing_ok=True)
            input_link.symlink_to(video_paths[vid].resolve())
            input_links[vid] = str(input_link)
        for r, c, vid, name, cond, key, cache_path in work:
            clip_dir = run_dir / row_labels[r]
            dst = clip_dir / f"{row_labels[r]}_{name}.mp4"
            dst.unlink(missing_ok=True)
            shutil.copy2(cache_path, dst)
            per_cell_paths[f"{vid}/{name}"] = str(dst)

    cell_records = []
    for r, c, vid, name, cond, key, cache_path in work:
        preset = PRESETS[name]
        cell_records.append({
            "row": r,
            "col": c,
            "row_label": row_labels[r],
            "col_label": col_labels[c],
            "video_id": vid,
            "video_input_path": str(video_paths[vid]),
            "preset": name,
            "prompt": preset.prompt,
            "negative_prompt": preset.negative_prompt,
            "conditioning_scale": cond,
            "guidance": guidance,
            "steps": steps,
            "seed": seed,
            "dtype": dtype,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cache_key": key,
            "cache_path": str(cache_path),
            "output_path": per_cell_paths.get(f"{vid}/{name}"),
        })

    meta = {
        "timestamp_utc": stamp,
        "video_ids": list(video_ids),
        "preset_names": list(preset_names),
        "col_labels": col_labels,
        "row_labels": row_labels,
        "gpu_ids": list(gpu_ids),
        "model_path": str(model_path),
        "shared_params": {
            "guidance": guidance,
            "steps": steps,
            "seed": seed,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "dtype": dtype,
        },
        "conditioning_scale_input": (
            conditioning_scale
            if not isinstance(conditioning_scale, dict)
            else dict(conditioning_scale)
        ),
        "conditioning_scale_resolved": cond_by_preset,
        "grid": {
            "cell_w": cell_w, "cell_h": cell_h,
            "header_h": header_h, "label_w": label_w,
        },
        "cache": {
            "cache_dir": str(CACHE_DIR),
            "hits": n_hit, "misses": n_miss, "total": len(work),
        },
        "input_links": input_links,
        "cells": cell_records,
        "load_time_s": round(load_time, 2),
        "run_time_s": round(run_time, 2),
        "output": str(grid_path),
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )

    print(f"\nrun_dir : {run_dir}")
    print(f"grid    : {grid_path}")
    if per_cell_paths:
        print(f"per-clip: {len(input_links)} folders × ({len(preset_names)} edits + input.mp4)")
    print(f"meta    : {run_dir / 'meta.json'}")
    print(f"timing  : load {load_time:.1f}s + infer {run_time:.1f}s "
          f"({run_time / 60:.1f} min)")

    return {
        "run_dir": run_dir,
        "grid_path": grid_path,
        "per_cell_paths": per_cell_paths,
        "meta": meta,
    }
