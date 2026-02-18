#
# Copyright 2025 Splunk Inc.
#
# Modified by Allen Baranov to work with Orion.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Minimal inference script for Cisco's multi-resolution TimesFM.

Stripped to essentials - just load weights and make predictions.
"""
import os

import numpy as np
import torch
from huggingface_hub import snapshot_download
from timesfm import pytorch_patched_decoder as ppd
from torch import nn

from context_builder import build_multi_resolution_context


class MinimalCiscoModel(ppd.PatchedTimeSeriesDecoder):
    """Multi-resolution TimesFM with resolution embeddings and special token."""

    def __init__(self, config):
        super().__init__(config)

        self.multi_resolution = nn.Embedding(2, config.hidden_size)
        self.special_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        nn.init.normal_(self.special_token, mean=0.0, std=0.02)

    def reverse_transform_segments(self, outputs, stats_list, indices_list):
        """Apply per-segment denormalization.

        Args:
            outputs: [B, N, H, Q] predictions
            stats_list: [(mu, sigma), ...] for each segment
            indices_list: [(start, end), ...] token ranges for each segment

        Returns:
            Denormalized outputs with coarse positions using coarse stats,
            fine positions using fine stats.
        """
        B, N, H, Q = outputs.shape
        device = outputs.device
        dtype = outputs.dtype

        if len(indices_list) == 0:
            return outputs

        # Build segment masks
        starts = torch.tensor([s for (s, _) in indices_list], device=device)
        ends = torch.tensor([e for (_, e) in indices_list], device=device)
        S = starts.shape[0]  # number of segments

        # Stack stats [B, S]
        mus = torch.stack([mu.to(dtype) for (mu, _) in stats_list], dim=1)
        sigmas = torch.stack([sigma.to(dtype) for (_, sigma) in stats_list], dim=1)

        # Create mask for each segment: [S, N]
        posN = torch.arange(N, device=device)
        seg_mask = ((posN.unsqueeze(0) >= starts.unsqueeze(1)) &
                    (posN.unsqueeze(0) < ends.unsqueeze(1)))

        # Broadcast to [1, S, N, 1, 1]
        seg_mask = seg_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(dtype)
        mus_b = mus.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, S, 1, 1, 1]
        sigmas_b = sigmas.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Aggregate per-position: [B, N, 1, 1]
        mu_map = (mus_b * seg_mask).sum(dim=1)
        sigma_map = (sigmas_b * seg_mask).sum(dim=1)

        # For uncovered positions, use identity transform (sigma=1, mu=0)
        covered = (seg_mask.sum(dim=1) > 0).to(dtype)
        sigma_map = sigma_map + (1.0 - covered).expand(B, -1, -1, -1)

        return outputs * sigma_map + mu_map


def normalize(context, pad_mask):
    """Normalize context, ignoring padded values."""
    eps = 1e-8
    valid = 1.0 - pad_mask  # 1 for real data, 0 for padding
    count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)

    mean = (context * valid).sum(dim=1, keepdim=True) / count
    var = (((context - mean) * valid) ** 2).sum(dim=1, keepdim=True) / count
    std = var.sqrt()

    normalized = (context - mean) / (std + eps)
    normalized = normalized * valid
    normalized = torch.clamp(normalized, -1000, 1000)

    return normalized, mean, std


def preprocess_series(series, agg_factor=60):
    """Convert raw series to normalized coarse/fine contexts.

    Args:
        series: 1D numpy array
        agg_factor: Aggregation factor (default 60 = minute->hour)

    Returns:
        coarse_ctx, coarse_pad, fine_ctx, fine_pad,
        coarse_offset, coarse_scale, fine_offset, fine_scale
    """
    coarse_list, fine_list = build_multi_resolution_context(
        series,
        agg_factor=agg_factor,
        max_coarse_ctx=512,
        max_fine_ctx=512
    )

    coarse = torch.tensor(coarse_list, dtype=torch.float32)
    fine = torch.tensor(fine_list, dtype=torch.float32)

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

    coarse_norm, coarse_offset, coarse_scale = normalize(
        coarse.unsqueeze(0), coarse_mask.unsqueeze(0))
    fine_norm, fine_offset, fine_scale = normalize(fine.unsqueeze(0), fine_mask.unsqueeze(0))

    return (
        coarse_norm.squeeze(0),
        coarse_mask,
        fine_norm.squeeze(0),
        fine_mask,
        float(coarse_offset.item()),
        float(coarse_scale.item()),
        float(fine_offset.item()),
        float(fine_scale.item())
    )


class CiscoInference:
    """Minimal Cisco TimesFM inference."""

    def __init__(self, repo_id="cisco-ai/cisco-time-series-model-1.0-preview"):
        """Load model from HuggingFace."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Download checkpoint
        print(f"Downloading from {repo_id}...")
        checkpoint_dir = snapshot_download(repo_id)
        checkpoint_path = os.path.join(checkpoint_dir, "torch_model.pt")

        config = ppd.TimesFMConfig(
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

        self.model = MinimalCiscoModel(config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.config = config

    def forecast(self, series_list, horizon_len=128, agg_factor=60):
        """
        Forecast time series.

        Args:
            series_list: List of 1D numpy arrays
            horizon_len: Steps to forecast (default 128)
            agg_factor: Aggregation factor (default 60)

        Returns:
            List of dicts with 'mean' and 'quantiles'
        """
        # Preprocess all series
        preprocessed = []
        for series in series_list:
            if not isinstance(series, np.ndarray):
                series = np.array(series, dtype=np.float32)
            preprocessed.append(preprocess_series(series, agg_factor))

        results = []

        with torch.no_grad():
            for coarse, c_pad, fine, f_pad, c_offset, c_scale, f_offset, f_scale in preprocessed:
                coarse = coarse.unsqueeze(0).unsqueeze(-1).to(self.device)
                c_pad = c_pad.unsqueeze(0).unsqueeze(-1).to(self.device)
                fine = fine.unsqueeze(0).unsqueeze(-1).to(self.device)
                f_pad = f_pad.unsqueeze(0).unsqueeze(-1).to(self.device)

                # Preprocess through model
                coarse_proc, c_pad_proc, stats_coarse, _ = self.model._preprocess_input(
                    coarse, c_pad)
                fine_proc, f_pad_proc, stats_fine, _ = self.model._preprocess_input(fine, f_pad)

                # Concatenate with special token
                spec_token = self.model.special_token.expand(1, 1, -1)
                spec_pad = torch.zeros(1, 1, device=self.device)

                model_input = torch.cat([coarse_proc, spec_token, fine_proc], dim=1)
                padding = torch.cat([c_pad_proc, spec_pad, f_pad_proc], dim=1)

                # Add resolution embeddings
                N_coarse = coarse_proc.shape[1]
                N_fine = fine_proc.shape[1]

                res_ids = torch.cat([
                    torch.zeros(N_coarse, dtype=torch.long),
                    torch.zeros(1, dtype=torch.long),
                    torch.ones(N_fine, dtype=torch.long)
                ]).unsqueeze(0).to(self.device)

                res_emb = self.model.multi_resolution(res_ids)
                model_input = model_input + res_emb

                # Add frequency embedding
                freq_emb = self.model.freq_emb(torch.zeros(
                    1, 1, dtype=torch.long, device=self.device))
                model_input = model_input + freq_emb

                # Forward through transformer
                output = self.model.stacked_transformer(model_input, padding)

                # Project to predictions
                predictions = self.model.horizon_ff_layer(output)

                # Reshape to [B, N, H, Q]
                num_outputs = len(self.config.quantiles) + 1
                predictions = predictions.view(1, -1, self.config.horizon_len, num_outputs)

                # Apply per-segment reverse transform
                # Build stats using the ORIGINAL normalization stats (not model's internal ones)
                stats_list = [
                    (torch.tensor([c_offset], device=self.device),
                     torch.tensor([c_scale], device=self.device)),
                    (torch.tensor([f_offset], device=self.device),
                     torch.tensor([f_scale], device=self.device))
                ]
                indices_list = [
                    (0, N_coarse),  # Coarse segment
                    (N_coarse + 1, N_coarse + 1 + N_fine)  # Fine segment (skip special token)
                ]

                predictions = self.model.reverse_transform_segments(
                    predictions, stats_list, indices_list
                )

                # Extract prediction from last fine token
                pred = predictions[0, -1, :horizon_len, :].cpu().numpy()

                # Build result
                result = {
                    'mean': pred[:, 0],
                    'quantiles': {
                        str(q): pred[:, i + 1]
                        for i, q in enumerate(self.config.quantiles)
                    }
                }
                results.append(result)

        return results


if __name__ == "__main__":
    print("Cisco Multi-Resolution TimesFM - Minimal")
    print("=" * 50)

    model = CiscoInference()

    np.random.seed(42)
    sample_series = np.cumsum(np.random.randn(10000)) + 100

    results = model.forecast([sample_series], horizon_len=128)

    result = results[0]
    print(f"\nMean forecast (first 10): {result['mean'][:10]}")
    print(f"Available quantiles: {list(result['quantiles'].keys())}")
    print("\n✓ Done!")
