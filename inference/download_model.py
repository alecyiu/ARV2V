#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface_hub>=0.27",
# ]
# ///
"""Download the Wan 2.1 VACE 1.3B snapshot to checkpoints/Wan2.1-VACE-1.3B/.

Run once before invoking v2v_recon.py. Skips files that are already present
(huggingface_hub handles resumption).

Usage:
    uv run inference/download_model.py
    uv run inference/download_model.py --repo Wan-AI/Wan2.1-VACE-1.3B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sibling `config.py` importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CHECKPOINTS_ROOT, WAN_VACE_1_3B_LOCAL, WAN_VACE_1_3B_REPO  # noqa: E402

from huggingface_hub import snapshot_download  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=WAN_VACE_1_3B_REPO,
                        help=f"HF repo id (default: {WAN_VACE_1_3B_REPO})")
    parser.add_argument("--local-dir", default=str(WAN_VACE_1_3B_LOCAL),
                        help=f"Where to put the snapshot (default: {WAN_VACE_1_3B_LOCAL})")
    parser.add_argument("--allow-patterns", nargs="*", default=None,
                        help="Optional file patterns to limit the download "
                             "(e.g. '*.safetensors' '*.json' '*.txt')")
    args = parser.parse_args()

    CHECKPOINTS_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo} -> {args.local_dir}")
    path = snapshot_download(
        repo_id=args.repo,
        local_dir=args.local_dir,
        allow_patterns=args.allow_patterns,
    )
    print(f"Done. Snapshot at: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
