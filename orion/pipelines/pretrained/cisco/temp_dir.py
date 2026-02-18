import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

with tempfile.TemporaryDirectory() as tmpdir:
    repo_path = Path(tmpdir) / "cisco-time-series-model"

    subprocess.check_call([
        "git", "clone", "--depth", "1",
        "https://github.com/splunk/cisco-time-series-model.git",
        str(repo_path)
    ])

    preview_path = repo_path / "1.0-preview"
    sys.path.insert(0, str(preview_path))

    print("\nCloned repo location:")
    print(repo_path.resolve())
    print("\nPreview path:")

    from modeling import CiscoTsmMR, TimesFmHparams, TimesFmCheckpoint

    rng = np.random.default_rng(42)

    T = 512 * 60
    hours = (T + 59) // 60
    k = np.arange(hours, dtype=np.float32)
    h = (80 + 0.1 * k) * (1 + 0.25 * np.sin(2 * np.pi * k / 24))
    t = np.arange(T, dtype=np.float32)

    input_series_1 = (
        h[(t // 60).astype(int)]
        * (1 + 0.05 * np.sin(2 * np.pi * t / 30))
        + rng.normal(0, 0.4, size=T)
    )

    hparams = TimesFmHparams(
        num_layers=50,
        use_positional_embedding=False,
        backend="gpu" if torch.cuda.is_available() else "cpu",
    )

    ckpt = TimesFmCheckpoint(
        huggingface_repo_id="cisco-ai/cisco-time-series-model-1.0-preview"
    )

    model = CiscoTsmMR(
        hparams=hparams,
        checkpoint=ckpt,
        use_resolution_embeddings=True,
        use_special_token=True,
    )

    # Inference
    forecast_preds = model.forecast(input_series_1, horizon_len=128)

    print("Mean forecast (first 5):", forecast_preds[0]["mean"][:5])

print("Temporary repo cleaned up.")
