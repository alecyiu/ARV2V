"""Shared paths, model identity, and default generation parameters.

Importable from both `download_model.py` and `v2v_recon.py` (and notebooks),
so any later change to a path or default lives in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/alec/ARV2V")

# Source data (read-only)
WAYMO_ROOT = PROJECT_ROOT / "waymo"
WAYMO_VIDEOS_DIR = WAYMO_ROOT / "videos"
WAYMO_TRAIN_META = WAYMO_ROOT / "train" / "metadata_vace_general.csv"

# Model snapshots and run outputs
CHECKPOINTS_ROOT = PROJECT_ROOT / "checkpoints"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
V2V_RECON_OUTPUTS = OUTPUTS_ROOT / "v2v_recon"

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------
# Hugging Face repo for the diffusers-formatted Wan 2.1 VACE 1.3B model.
# (If diffusers loading fails, try the non-diffusers repo "Wan-AI/Wan2.1-VACE-1.3B"
#  and run the official Wan inference code instead.)
WAN_VACE_1_3B_REPO = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
WAN_VACE_1_3B_LOCAL = CHECKPOINTS_ROOT / "Wan2.1-VACE-1.3B"

# ---------------------------------------------------------------------------
# Generation defaults for the 1.3B model
# ---------------------------------------------------------------------------
# VACE-1.3B is trained at 480p, 49 or 81 frames, 16 fps. Our Waymo clips are
# 49 frames at 10 fps. We feed all 49 frames as the control video; the output
# is written at the *source* fps so input/output align in wall-clock time
# during side-by-side viewing.
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832
DEFAULT_NUM_FRAMES = 49
DEFAULT_FPS_OUT = 10  # matches Waymo source fps, even though VACE is "16 fps"
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_PROMPT = "The video depicts a view of a street"  # from train metadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def video_path_for(video_id: str) -> Path:
    """Resolve a short id like '044' or 'waymo_000044' (or full filename) to an absolute path."""
    name = video_id
    if not name.endswith(".mp4"):
        if not name.startswith("waymo_"):
            # accept '44', '044', '000044'
            name = f"waymo_{int(name):06d}"
        name = f"{name}.mp4"
    return WAYMO_VIDEOS_DIR / name
