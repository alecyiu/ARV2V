# V2V reconstruction with Wan 2.1 VACE 1.3B

Single-clip video-to-video reconstruction. Feeds one Waymo training clip
through `Wan2.1-VACE-1.3B` as the control video and writes the reconstructed
mp4 plus a side-by-side comparison.

## Layout

```
ARV2V/
├── checkpoints/Wan2.1-VACE-1.3B/        # model snapshot (gitignored)
├── inference/
│   ├── config.py                        # paths + defaults (edit here)
│   ├── download_model.py                # one-shot HF snapshot download
│   ├── v2v_recon.py                     # the inference script
│   └── README.md                        # this file
└── outputs/v2v_recon/<UTC-stamp>_<vid>/  # per-run artefacts (gitignored)
    ├── input.mp4                        # symlink to source
    ├── output.mp4                       # reconstructed video
    ├── side_by_side.mp4                 # control | output
    └── meta.json                        # all params + timing
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
