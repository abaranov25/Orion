#!/usr/bin/env python3
"""
Minimal end-to-end demo:
1) Build fine (xf) and coarse (xc) multiresolution contexts from a single 1-min series
2) Load Cisco checkpoint from Hugging Face
3) Forecast

Notes:
- Cisco model expects (xc, xf) where xc has 60x coarser resolution than xf, both length<=512,
  aligned on the right. :contentReference[oaicite:0]{index=0}
- This script works even if you're NOT inside the cloned repo: it clones the repo into ./_ctsm_src
  just to import `modeling.py`, while weights come from Hugging Face.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_URL = "https://github.com/splunk/cisco-time-series-model.git"
REPO_DIR = Path("_ctsm_src")
MODEL_PY_DIR = REPO_DIR / "1.0-preview"  # contains modeling.py in their repo layout
HF_REPO_ID = "cisco-ai/cisco-time-series-model-1.0-preview"


def ensure_modeling_importable():
    if not MODEL_PY_DIR.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
    sys.path.insert(0, str(MODEL_PY_DIR))


def make_multires_context(x_1min: np.ndarray, ratio: int = 60, L: int = 512):
    """
    x_1min: 1D array at fine resolution (e.g., 1-minute samples)
    Returns: (xc, xf), both float32, each length L, aligned on the right.
      xf = last L fine points
      xc = last L coarse points computed from last (L*ratio) fine points by taking the LAST
           value in each ratio-sized block (right-aligned).
    """
    x = np.asarray(x_1min, dtype=np.float32).reshape(-1)
    if x.size < L * ratio:
        raise ValueError(f"Need at least {L*ratio} fine points, got {x.size}")

    fine_tail = x[-L:]  # xf
    tail = x[-L * ratio:]  # last L hours worth (if ratio=60)
    blocks = tail.reshape(L, ratio)  # already right-aligned by construction
    coarse = blocks[:, -1]  # "last value in block" coarse sampler

    return coarse.astype(np.float32), fine_tail.astype(np.float32)


def main():
    ensure_modeling_importable()

    # Import AFTER sys.path tweak
    from modeling import CiscoTsmMR, TimesFmHparams, TimesFmCheckpoint

    # ----- make a toy 1-min signal with daily-ish structure -----
    rng = np.random.default_rng(0)
    T = 512 * 60  # exactly enough for xc/xf with ratio=60, L=512
    t = np.arange(T, dtype=np.float32)

    # slow trend + daily sinusoid + faster wiggle + noise
    x_1min = (
        50.0
        + 0.0005 * t
        + 5.0 * np.sin(2 * np.pi * t / (60 * 24))
        + 1.0 * np.sin(2 * np.pi * t / 30)
        + rng.normal(0, 0.5, size=T).astype(np.float32)
    )

    xc, xf = make_multires_context(x_1min, ratio=60, L=512)

    # ----- load Cisco weights from Hugging Face -----
    hparams = TimesFmHparams(
        num_layers=50,
        use_positional_embedding=False,
        backend="gpu" if torch.cuda.is_available() else "cpu",
    )
    ckpt = TimesFmCheckpoint(huggingface_repo_id=HF_REPO_ID)

    model = CiscoTsmMR(
        hparams=hparams,
        checkpoint=ckpt,
        use_resolution_embeddings=True,
        use_special_token=True,
    )

    # ----- forecast (interpreted at fine resolution) -----
    preds = model.forecast((xc, xf), horizon_len=128)
    mean_forecast = preds[0]["mean"]  # (128,)

    print("xc shape:", xc.shape, "xf shape:", xf.shape)
    print("mean_forecast[:5]:", mean_forecast[:5])


if __name__ == "__main__":
    main()
