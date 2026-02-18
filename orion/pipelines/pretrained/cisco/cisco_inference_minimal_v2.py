"""
Minimal inference script for Cisco's multi-resolution TimesFM.

Same structure as your script, but fixes the real bugs:
- uses the stats returned by `_preprocess_input` and applies the per-segment reverse-transform
- drops the special-token position before selecting the fine token
- selects the fine token by patch-count logic (not `-1`)
- avoids device footguns for the special token
- actually applies strip/interp preprocessing you imported
"""
import os

import numpy as np
import torch
from huggingface_hub import snapshot_download
from timesfm import pytorch_patched_decoder as ppd
from timesfm.timesfm_base import linear_interpolation, strip_leading_nans
from torch import nn

from .context_builder import build_multi_resolution_context


class MinimalCiscoModel(ppd.PatchedTimeSeriesDecoder):
    """Multi-resolution TimesFM with resolution embeddings and special token."""

    def __init__(self, config):
        super().__init__(config)
        self.multi_resolution = nn.Embedding(2, config.hidden_size)
        self.special_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    @staticmethod
    def reverse_transform_two_segments(outputs, stats_coarse, stats_fine, N_coarse, N_fine):
        """
        Faster + simpler than mask-mapping.
        outputs: [B, N_total, H, Q]
        stats_*: (mu, sigma) from _preprocess_input (shapes vary: [B], [B,1], [B,1,1], ...)
        Applies y = y*sigma + mu on:
          - coarse tokens: [0 : N_coarse)
          - fine tokens:   [N_coarse+1 : N_coarse+1+N_fine)   (skips special token at N_coarse)
        """
        B = outputs.shape[0]
        dtype = outputs.dtype

        mu_c = stats_coarse[0].to(dtype).reshape(B, -1)[:, 0]
        sg_c = stats_coarse[1].to(dtype).reshape(B, -1)[:, 0]
        mu_f = stats_fine[0].to(dtype).reshape(B, -1)[:, 0]
        sg_f = stats_fine[1].to(dtype).reshape(B, -1)[:, 0]

        # coarse segment
        outputs[:, 0:N_coarse, :, :] = outputs[:, 0:N_coarse, :, :] * \
            sg_c.view(B, 1, 1, 1) + mu_c.view(B, 1, 1, 1)

        # fine segment (skip separator token at index N_coarse)
        s1 = N_coarse + 1
        e1 = N_coarse + 1 + N_fine
        outputs[:, s1:e1, :, :] = outputs[:, s1:e1, :, :] * \
            sg_f.view(B, 1, 1, 1) + mu_f.view(B, 1, 1, 1)

        return outputs


def preprocess_series(series, agg_factor=60):
    """
    Return RAW (not normalized) padded contexts + pad masks.
    Let timesfm `_preprocess_input` do its own normalization and return stats.
    """
    series = np.asarray(series, dtype=np.float32)
    if not np.isfinite(series).all():
        series = np.where(np.isfinite(series), series, np.nan)
    series = strip_leading_nans(series)
    series = linear_interpolation(series)

    coarse_list, fine_list = build_multi_resolution_context(
        series,
        agg_factor=agg_factor,
        max_coarse_ctx=512,
        max_fine_ctx=512
    )

    coarse = torch.tensor(coarse_list, dtype=torch.float32)
    fine = torch.tensor(fine_list, dtype=torch.float32)

    # LEFT pad to 512 with pad masks (1=pad, 0=real)
    if len(coarse) < 512:
        pad_len = 512 - len(coarse)
        coarse = torch.cat([torch.zeros(pad_len), coarse])
        coarse_mask = torch.cat([torch.ones(pad_len), torch.zeros(len(coarse_list))])
    else:
        coarse = coarse[-512:]
        coarse_mask = torch.zeros(512)

    if len(fine) < 512:
        pad_len = 512 - len(fine)
        fine = torch.cat([torch.zeros(pad_len), fine])
        fine_mask = torch.cat([torch.ones(pad_len), torch.zeros(len(fine_list))])
    else:
        fine = fine[-512:]
        fine_mask = torch.zeros(512)

    return coarse, coarse_mask, fine, fine_mask


class CiscoInference:
    """Minimal Cisco TimesFM inference."""

    def __init__(self, repo_id="cisco-ai/cisco-time-series-model-1.0-preview"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Downloading from {repo_id}...")
        checkpoint_dir = snapshot_download(repo_id)
        checkpoint_path = os.path.join(checkpoint_dir, "torch_model.pt")

        self.config = ppd.TimesFMConfig(
            num_layers=50,
            num_heads=16,
            hidden_size=1280,
            intermediate_size=1280,
            patch_len=32,
            horizon_len=128,
            head_dim=80,
            quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            use_positional_embedding=False,
        )

        self.model = MinimalCiscoModel(self.config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def forecast(self, series_list, horizon_len=128, agg_factor=60):
        assert horizon_len <= self.config.horizon_len, (
            f"horizon_len must be <= model horizon {self.config.horizon_len}"
        )

        results = []

        with torch.no_grad():
            for series in series_list:
                if not isinstance(series, np.ndarray):
                    series = np.array(series, dtype=np.float32)

                # stream preprocessing (no preprocessed list)
                coarse_raw, c_padmask, fine_raw, f_padmask = preprocess_series(
                    series, agg_factor=agg_factor)

                coarse = coarse_raw.unsqueeze(0).unsqueeze(-1).to(self.device)   # [1,512,1]
                c_pad = c_padmask.unsqueeze(
                    0).unsqueeze(-1).to(self.device)    # [1,512,1] or [1,512]
                fine = fine_raw.unsqueeze(0).unsqueeze(-1).to(self.device)
                f_pad = f_padmask.unsqueeze(0).unsqueeze(-1).to(self.device)

                coarse_proc, c_pad_proc, stats_coarse, _ = self.model._preprocess_input(
                    coarse, c_pad)
                fine_proc, f_pad_proc, stats_fine, _ = self.model._preprocess_input(fine, f_pad)

                B = coarse_proc.shape[0]   # 1
                N_coarse = coarse_proc.shape[1]
                N_fine = fine_proc.shape[1]
                D = coarse_proc.shape[2]

                # padding must be 2D [B,N] for stacked_transformer
                if c_pad_proc.ndim == 3:
                    c_pad_proc = c_pad_proc.squeeze(-1)
                if f_pad_proc.ndim == 3:
                    f_pad_proc = f_pad_proc.squeeze(-1)

                # special token (already on device; no redundant .to())
                spec_token = self.model.special_token.expand(B, 1, D)
                spec_pad = torch.zeros(B, 1, device=self.device, dtype=c_pad_proc.dtype)

                model_input = torch.cat([coarse_proc, spec_token, fine_proc], dim=1)
                padding = torch.cat([c_pad_proc, spec_pad, f_pad_proc], dim=1)

                # resolution embeddings
                res_ids = torch.cat([
                    torch.zeros(N_coarse, dtype=torch.long, device=self.device),
                    torch.zeros(1, dtype=torch.long, device=self.device),
                    torch.ones(N_fine, dtype=torch.long, device=self.device),
                ]).unsqueeze(0).expand(B, -1)

                model_input = model_input + self.model.multi_resolution(res_ids)

                # frequency embedding (constant 0)
                freq = torch.zeros(B, 1, dtype=torch.long, device=self.device)
                model_input = model_input + self.model.freq_emb(freq)

                output = self.model.stacked_transformer(model_input, padding)
                logits = self.model.horizon_ff_layer(output)

                num_outputs = len(self.config.quantiles) + 1
                preds = logits.view(B, -1, self.config.horizon_len,
                                    num_outputs)  # [B, N_total, H, Q]

                # faster reverse transform (two slices)
                preds = self.model.reverse_transform_two_segments(
                    preds, stats_coarse, stats_fine, N_coarse, N_fine
                )

                fine_token_idx = N_coarse + N_fine
                token_pred = preds[:, fine_token_idx, :horizon_len, :]

                token_np = token_pred[0].cpu().numpy()
                results.append({
                    "mean": token_np[:, 0],
                    "quantiles": {str(q): token_np[:, i + 1] for i, q in enumerate(self.config.quantiles)},
                })

        return results


if __name__ == "__main__":
    print("Cisco Multi-Resolution TimesFM - Minimal (fixed)")
    print("=" * 50)

    model = CiscoInference()

    np.random.seed(42)
    sample_series = np.cumsum(0.1 + 0.01 * np.random.randn(10000)).astype(np.float32) + 100.0

    results = model.forecast([sample_series[:-128]], horizon_len=128)
    result = results[0]

    print(np.mean(np.abs(result["mean"] - sample_series[-128:])))
    print(f"\nMean forecast (first 10): {result['mean']}")
    print(sample_series[-128:])
    print(f"Available quantiles: {list(result['quantiles'].keys())}")
    print("\n✓ Done!")
