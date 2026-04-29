# ARV2V 

Auto regressive V2V — Wan 2.1 VACE 1.3B inference pipeline for Waymo
front-camera (camera_id=0) clips.

## Layout

```
ARV2V/
├── inference/
│   ├── config.py              shared paths and generation defaults
│   ├── download_model.py      one-time: pull VACE 1.3B from HF
│   └── v2v_recon.py           single-clip V2V reconstruction / editing
├── analysis/
│   ├── check_dataset.ipynb    dataset coverage check (camera_id=0)
│   ├── check_dataset.py       stdlib coverage summary
│   └── copy_camera0_videos.py copy camera_id=0 train+val from /miele
├── waymo/                     local mirror (populated by copy script)
├── checkpoints/               model snapshots
└── outputs/v2v_recon/         per-run output folders
```

## Setup

Requires Python ≥3.12 and a CUDA 12.6-capable NVIDIA driver (the project pins
torch to the cu126 wheel index — see `pyproject.toml`).

```bash
cd ~/ARV2V
uv sync
```

Verify the GPU is visible:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Download the VACE 1.3B model once:

```bash
uv run inference/download_model.py
```

## 1. Download the dataset from `/miele/`

The canonical Waymo dataset lives on NFS at `/miele/liory/datasets/waymo/`.
Inference only needs the **forward camera (`camera_id=0`)** subset:
**12,304 train + 201 val** clips.

```bash
# preview (no writes)
uv run analysis/copy_camera0_videos.py --dry-run

# copy (parallel, NFS-aware; resumable)
uv run analysis/copy_camera0_videos.py
```

What it does:

1. Reads each split's metadata CSV from `/miele/.../{train,val}/`.
2. Filters rows where `camera_id == 0`.
3. Copies the referenced mp4s to `~/ARV2V/waymo/` mirroring the source layout
   (train videos go to `videos/`, val videos to `val/videos/`).
4. Copies the metadata CSVs alongside.

Notes:

- `/miele` is NFS — copies are latency-bound. The script defaults to
  `--workers 16`; bump to `--workers 32` if the NFS server tolerates it.
- Resumable: skips files already present locally without re-stating the
  source. Pass `--verify` once after an interrupted run to size-check
  existing files (catches partial copies from older runs).
- New copies write to `<name>.tmp` then atomic-rename, so future runs never
  see partial files.
- `--splits train val` by default; pass `--splits val` to do just one.

Verify coverage afterwards:

```bash
uv run analysis/check_dataset.py
```

Should print `12,304 / 12,304` for `camera_id=0` (train) once the copy
finishes.

## 2. Generate an edited video in one command

After setup + dataset copy + model download, a full V2V edit run is one
command. Example: turn `waymo_000000.mp4` into a snowy version on GPU 3.

```bash
uv run python inference/v2v_recon.py \
  --video-id 0 \
  --gpu 3 \
  --conditioning-scale 0.4 \
  --guidance 8.0 \
  --steps 50 \
  --prompt "A heavy winter snowstorm view from a forward-facing dashcam. \
The street is completely buried under deep fresh snow, several inches thick, \
white and unbroken except for parallel tire ruts. Every parked car is heavily \
covered in snow on the roof, hood, and trunk. Sidewalks, lawns, rooftops, \
fences, and traffic signs are fully blanketed in white. Bare tree branches \
are caked with snow and sag heavily. The sky is a flat overcast pale gray. \
Cold diffuse winter light, no shadows. Visible falling snowflakes drift \
through the air. Photorealistic winter footage."
```

This produces a fresh folder under `outputs/v2v_recon/<UTC>_<clip>/`
containing:

- `input.mp4` — symlink to the source
- `output.mp4` — generated 49 frames at the source fps
- `side_by_side.mp4` — input | output, useful for A/B viewing
- `meta.json` — every flag, prompt, timing, and GPU used

### CLI essentials

| flag | default | role |
|---|---|---|
| `--video-id` | required | `0`, `000044`, or `waymo_000044` |
| `--prompt` | "The video depicts a view of a street" | UMT5-XXL conditioning |
| `--negative-prompt` | automotive negative (see `config.py`) | what to steer away from |
| `--conditioning-scale` | 1.0 | source-video grip; **lower = more editing** |
| `--guidance` | 5.0 | CFG strength on (positive − negative) prompt |
| `--steps` | 30 | denoising steps; raise for big edits |
| `--gpu` | 0 | CUDA device index |
| `--no-side-by-side` | off | skip the comparison render |

### Tuning intuition

- **Reconstruct only** (no edit): default `--conditioning-scale 1.0`. Output
  closely tracks the source.
- **Light style/color edit** (e.g. golden hour, rainy): `0.6–0.7`.
- **Weather/season change** (e.g. snow): `0.35–0.45` plus `--guidance 7–9`
  and `--steps 40–50`.
- **Strong content edit** (e.g. fictional scene): `0.2–0.3`. Below ~0.25 the
  camera path and lane structure start breaking.

The CFG (`--guidance`) and conditioning scale work in opposite directions:
raising guidance amplifies the prompt vs. unconditional, while lowering
conditioning weakens the source vs. the prompt. For big edits, push both.

## How the pipeline works (one-pager)

`v2v_recon.py` calls `WanVACEPipeline` from diffusers, which:

1. **Tokenizes + encodes the prompt** through UMT5-XXL → `(1, 512, 4096)`
   embedding (used as cross-attention K/V in every denoising step).
2. **VAE-encodes the source video** as the VACE control signal (the
   "reactive" channel, since `mask=None` defaults to all-ones).
3. **Denoises** `num_inference_steps` times via `UniPCMultistepScheduler`,
   each step running the WanVACETransformer with text cross-attention and
   VACE control conditioning.
4. **VAE-decodes** the final latents → output frames.

So the prompt and the source video are *both* control signals competing for
the model's attention. `--conditioning-scale` is the lever that re-weights
their fight.
