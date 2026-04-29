# V2V reconstruction with Wan 2.1 VACE 1.3B

Video-to-video reconstruction and prompt-driven editing. Feeds Waymo dashcam
clips through `Wan2.1-VACE-1.3B` as the control video and writes the
reconstructed mp4s plus side-by-side or grid comparisons.

Three entry points share the same `utils/` helpers:

- `v2v_recon.py` — one-shot CLI for a single clip.
- `single_inference.ipynb` — interactive notebook for iterating on prompts/seeds against one clip.
- `grid_demo.ipynb` — 5×5 themed comparison: 5 ground-truth clips × (1 GT col + 4 themed edit cols).

## Layout

```
ARV2V/
├── checkpoints/Wan2.1-VACE-1.3B/        # model snapshot (gitignored)
├── inference/
│   ├── config.py                        # paths + generation defaults
│   ├── prompts.py                       # 4 themed PromptPreset entries (snow / mario_kart / cyberpunk / desert)
│   ├── download_model.py                # one-shot HF snapshot download
│   ├── v2v_recon.py                     # CLI entry point
│   ├── single_inference.ipynb           # one-clip interactive notebook
│   ├── grid_demo.ipynb                  # 5x5 themed grid notebook
│   ├── utils/                           # shared helpers (importable from all three entry points)
│   │   ├── video_io.py                  # read/write mp4, center-crop-resize, dtype coercion
│   │   ├── pipeline.py                  # WanVACEPipeline load + `infer_one`
│   │   ├── grid.py                      # `compose_grid_video` (PIL margins + tiled mp4)
│   │   └── cache.py                     # per-cell mp4 cache keyed by all generation params
│   └── README.md                        # this file
└── outputs/v2v_recon/
    ├── <UTC-stamp>_<vid>/               # per-clip run dir (CLI / single notebook)
    │   ├── input.mp4                    # symlink to source
    │   ├── output.mp4                   # reconstructed video
    │   ├── side_by_side.mp4             # control | output
    │   └── meta.json                    # all params + timing
    ├── <UTC-stamp>_grid_demo/           # grid run dir
    │   ├── grid.mp4                     # 5x5 tiled comparison with column/row labels
    │   └── meta.json                    # prompts, cache keys, params, timings
    └── _cache/cells/<sha1>.mp4          # per-cell cache; reused across grid reruns
```

## One-time setup

Install inference deps into the project venv:

```bash
cd /home/alec/ARV2V
source .venv/bin/activate
uv pip install \
    'torch==2.5.*' --index-url https://download.pytorch.org/whl/cu124
uv pip install \
    'diffusers>=0.34' transformers accelerate \
    huggingface_hub safetensors imageio[pyav] pillow
```

(Pin `torch` to a CUDA-12 wheel matching the host driver; the box has CUDA 12.6.)

Download the model snapshot:

```bash
python inference/download_model.py
# -> writes ~14-25 GB into checkpoints/Wan2.1-VACE-1.3B/
```

## Run a single clip

```bash
# pick a front-camera clip that exists on disk:
#   df.query('camera_id == 0 and file_present').head()
# e.g. waymo_000078

python inference/v2v_recon.py --video-id 78
```

Artefacts land in `outputs/v2v_recon/<UTC-timestamp>_waymo_000078/`.

Common knobs:

```bash
# explicit args (all optional; defaults live in config.py)
python inference/v2v_recon.py \
    --video-id waymo_000078 \
    --prompt "The video depicts a view of a street" \
    --height 480 --width 832 --num-frames 49 \
    --steps 30 --guidance 5.0 --seed 0 \
    --dtype bf16
```

## Grid demo

`grid_demo.ipynb` produces a 5×5 mp4 — 5 GT clips × (1 GT column + 4 themed
edit columns: snow / mario_kart / cyberpunk / desert). Defaults to `STEPS=50`
and ~80 min cold runtime; per-cell mp4s are cached at
`outputs/v2v_recon/_cache/cells/<sha1>.mp4` so reruns of the same parameters
finish in seconds. Edit the **Parameters** cell to scope down for smoke tests
(e.g. `VIDEO_IDS=["3"]`, `PRESET_NAMES=["snow"]`, `STEPS=4` for a ~30s sanity
check of the grid composition).

## Picking a GPU

The host has 10× RTX A6000 (48 GB). Pin to a specific card:

```bash
CUDA_VISIBLE_DEVICES=0 python inference/v2v_recon.py --video-id 78
```

VACE-1.3 B with `enable_model_cpu_offload()` peaks around 10–14 GB VRAM, so
any card with ≥20 GB free is fine. Check `nvidia-smi` first.

## Notes

- **Source fps is 10, VACE trains at 16 fps.** The script reads all 49 frames
  unchanged and writes the output at the source fps so a side-by-side plays
  in the same wall-clock time. Motion in the output may look slightly
  uncanny because the model assumed 16 fps internally — fine for a recon
  sanity check; revisit if quality is the bottleneck.
- **Aspect ratio.** Source frames are 864×576 or 1248×576; the script
  center-crops to 832×480 (VACE's native bucket).
- **Prompt.** All training rows have the same prompt
  `"The video depicts a view of a street"` — that's the default.
- The 14B variant (`Wan-AI/Wan2.1-VACE-14B-diffusers`) is a drop-in via
  `--model-path`, but expect ~3–6 minutes per clip on an idle A6000 and
  much longer with offload on a contested card.
