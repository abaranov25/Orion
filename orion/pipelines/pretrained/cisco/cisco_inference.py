#
# Copyright 2025 Splunk Inc.
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
Standalone inference script for Cisco's multi-resolution TimesFM model.

This script loads the Cisco model weights and implements inference WITHOUT
importing any vendored cisco_modeling code. It only uses:
- Base TimesFM from the timesfm package
- Context building functions from context_builder.py
- Custom preprocessing pipeline
"""
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download
# Base TimesFM imports (not vendored code)
from timesfm import pytorch_patched_decoder as ppd
from timesfm.timesfm_base import linear_interpolation, strip_leading_nans
from torch import nn

# Our context builder
from context_builder import build_multi_resolution_context


class MultiResolutionDecoder(ppd.PatchedTimeSeriesDecoder):
    """Multi-resolution extension of TimesFM decoder.

    Adds resolution embeddings and special token on top of base TimesFM.
    """

    def __init__(self, config, use_resolution_embeddings=True, use_special_token=True):
        super().__init__(config)
        self.use_resolution_embeddings = use_resolution_embeddings
        self.use_special_token = use_special_token

        # Add multi-resolution extensions
        if self.use_resolution_embeddings:
            self.multi_resolution = nn.Embedding(
                num_embeddings=2,
                embedding_dim=config.hidden_size
            )

        if self.use_special_token:
            self.special_token = nn.Parameter(
                torch.zeros(1, 1, config.hidden_size)
            )
            nn.init.normal_(self.special_token, mean=0.0, std=0.02)


def pad_or_truncate(
    ts: torch.Tensor,
    target_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Left-pad or truncate a time series to target length.

    Args:
        ts: 1D tensor
        target_len: desired length

    Returns:
        padded_ts: tensor of shape [target_len]
        pad_mask: tensor of shape [target_len], 1.0 for padded, 0.0 for real
    """
    if ts.ndim == 2 and ts.shape[-1] == 1:
        ts = ts.squeeze(-1)

    L = ts.shape[0]

    if L == target_len:
        return ts, torch.zeros_like(ts, dtype=torch.float32)

    if L > target_len:
        return ts[-target_len:], torch.zeros(target_len, dtype=torch.float32)

    # Left-pad
    pad_len = target_len - L
    padded = torch.cat([torch.zeros(pad_len, dtype=ts.dtype), ts], dim=0)
    pad_mask = torch.cat([
        torch.ones(pad_len, dtype=torch.float32),
        torch.zeros(L, dtype=torch.float32)
    ], dim=0)

    return padded, pad_mask


def normalize_with_pad(
    context: torch.Tensor,
    pad_mask: Optional[torch.Tensor] = None,
    clamp_range=(-1000, 1000)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Normalize context with padding awareness.

    Args:
        context: [B, T] or [B, T, 1]
        pad_mask: [B, T], 1.0 for padded, 0.0 for real

    Returns:
        normalized context, offset, scale, eps
    """
    eps = 1e-8

    if context.ndim == 3:
        context = context.squeeze(-1)

    if pad_mask is None:
        pad_mask = torch.zeros_like(context)

    valid = 1.0 - pad_mask
    count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)

    # Masked mean and std
    context_mean = (context * valid).sum(dim=1, keepdim=True) / count
    context_var = (((context - context_mean) * valid) ** 2).sum(dim=1, keepdim=True) / count
    context_std = context_var.sqrt()

    ctx_normalized = (context - context_mean) / (context_std + eps)
    ctx_normalized = ctx_normalized * valid
    ctx_normalized = torch.clamp(ctx_normalized, *clamp_range)

    return ctx_normalized, context_mean, context_std, eps


class CiscoInference:
    """Standalone inference class for Cisco's multi-resolution TimesFM."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        repo_id: str = "cisco-ai/cisco-time-series-model-1.0-preview",
        device: str = "auto",
        num_layers: int = 50,
        num_heads: int = 16,
        model_dims: int = 1280,
        input_patch_len: int = 32,
        output_patch_len: int = 128,
    ):
        """Initialize the inference model.

        Args:
            checkpoint_path: Local path to checkpoint. If None, downloads from HuggingFace.
            repo_id: HuggingFace repo ID for the model.
            device: Device to use ("cpu", "cuda", or "auto").
            num_layers: Number of transformer layers (50 for Cisco model).
            num_heads: Number of attention heads.
            model_dims: Model dimension.
            input_patch_len: Input patch length.
            output_patch_len: Output patch length.
        """
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Download checkpoint if needed
        if checkpoint_path is None:
            print(f"Downloading checkpoint from {repo_id}...")
            checkpoint_dir = snapshot_download(repo_id)
            checkpoint_path = os.path.join(checkpoint_dir, "torch_model.pt")

        # Build model config
        config = ppd.TimesFMConfig(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=model_dims,
            intermediate_size=model_dims,
            patch_len=input_patch_len,
            horizon_len=output_patch_len,
            head_dim=model_dims // num_heads,
            quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            use_positional_embedding=False,
        )

        # Create model with multi-resolution extensions
        self.model = MultiResolutionDecoder(
            config,
            use_resolution_embeddings=True,
            use_special_token=True
        )

        # Load weights
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        incompatible = self.model.load_state_dict(checkpoint, strict=False)

        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(f"Warning: Missing keys: {incompatible.missing_keys}")
            print(f"Warning: Unexpected keys: {incompatible.unexpected_keys}")

        self.model.to(self.device)
        self.model.eval()

        self.config = config
        print(f"✓ Model loaded on {self.device}")

    def preprocess_series(
        self,
        series: np.ndarray,
        agg_factor: int = 60,
        context_len_coarse: int = 512,
        context_len_fine: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        """Preprocess a single time series.

        Args:
            series: 1D numpy array
            agg_factor: Aggregation factor for coarse context
            context_len_coarse: Target length for coarse context
            context_len_fine: Target length for fine context

        Returns:
            Tuple of (coarse_tensor, coarse_pad, fine_tensor, fine_pad, offset, scale)
        """
        # Handle NaNs
        if not np.isfinite(series).all():
            series = np.where(np.isfinite(series), series, np.nan)
        series = strip_leading_nans(series)
        series = linear_interpolation(series)

        # Build multi-resolution contexts
        coarse_ctx, fine_ctx = build_multi_resolution_context(
            series,
            agg_factor=agg_factor,
            max_coarse_ctx=context_len_coarse,
            max_fine_ctx=context_len_fine
        )

        # Convert to tensors
        ctx_coarse = torch.tensor(coarse_ctx, dtype=torch.float32)
        ctx_fine = torch.tensor(fine_ctx, dtype=torch.float32)

        # Pad/truncate
        ctx_coarse_pad, mask_coarse = pad_or_truncate(ctx_coarse, context_len_coarse)
        ctx_fine_pad, mask_fine = pad_or_truncate(ctx_fine, context_len_fine)

        # Add batch dimension
        ctx_coarse_pad_b = ctx_coarse_pad.unsqueeze(0)
        mask_coarse_b = mask_coarse.unsqueeze(0)
        ctx_fine_pad_b = ctx_fine_pad.unsqueeze(0)
        mask_fine_b = mask_fine.unsqueeze(0)

        # Normalize
        norm_coarse, _, _, _ = normalize_with_pad(ctx_coarse_pad_b, mask_coarse_b)
        norm_fine, offset_fine, scale_fine, _ = normalize_with_pad(ctx_fine_pad_b, mask_fine_b)

        return (
            norm_coarse.squeeze(0),
            mask_coarse_b.squeeze(0),
            norm_fine.squeeze(0),
            mask_fine_b.squeeze(0),
            float(offset_fine.squeeze()),
            float(scale_fine.squeeze())
        )

    def forecast(
        self,
        inputs: List[np.ndarray],
        horizon_len: int = 128,
        agg_factor: int = 60,
        batch_size: int = 8,
    ) -> List[dict]:
        """Forecast multiple time series.

        Args:
            inputs: List of 1D numpy arrays (time series)
            horizon_len: Forecast horizon length
            agg_factor: Aggregation factor (e.g., 60 for min->hour)
            batch_size: Batch size for inference

        Returns:
            List of dicts with 'mean' and 'quantiles' keys
        """
        # Preprocess all series
        coarse_contexts = []
        coarse_pads = []
        fine_contexts = []
        fine_pads = []
        offsets = []
        scales = []

        for series in inputs:
            if not isinstance(series, np.ndarray):
                series = np.array(series, dtype=np.float32)

            c_ctx, c_pad, f_ctx, f_pad, offset, scale = self.preprocess_series(
                series, agg_factor=agg_factor
            )

            coarse_contexts.append(c_ctx.numpy())
            coarse_pads.append(c_pad.numpy())
            fine_contexts.append(f_ctx.numpy())
            fine_pads.append(f_pad.numpy())
            offsets.append(offset)
            scales.append(scale)

        # Batch inference
        N = len(inputs)
        all_predictions = []

        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)

                # Prepare batch
                batch_coarse = torch.as_tensor(
                    np.stack(coarse_contexts[start:end]),
                    dtype=torch.float32
                ).unsqueeze(-1).to(self.device)

                batch_coarse_pad = torch.as_tensor(
                    np.stack(coarse_pads[start:end]),
                    dtype=torch.float32
                ).unsqueeze(-1).to(self.device)

                batch_fine = torch.as_tensor(
                    np.stack(fine_contexts[start:end]),
                    dtype=torch.float32
                ).unsqueeze(-1).to(self.device)

                batch_fine_pad = torch.as_tensor(
                    np.stack(fine_pads[start:end]),
                    dtype=torch.float32
                ).unsqueeze(-1).to(self.device)

                # Simple forward pass (non-autoregressive for now)
                # For full autoregressive decoding, would need to implement the decode() method
                # This is a simplified version that does a single forward pass

                # Concatenate coarse and fine
                B = batch_coarse.shape[0]
                batch_coarse.shape[2]

                # Preprocess through model's _preprocess_input
                model_input_coarse, pad_coarse, stats_coarse, _ = self.model._preprocess_input(
                    batch_coarse, batch_coarse_pad
                )
                model_input_fine, pad_fine, stats_fine, _ = self.model._preprocess_input(
                    batch_fine, batch_fine_pad
                )

                # Add special token if enabled
                if self.model.use_special_token:
                    spec_tok = self.model.special_token.expand(B, 1, -1)
                    spec_pad = torch.zeros(B, 1, device=self.device, dtype=pad_coarse.dtype)
                    model_input = torch.cat(
                        [model_input_coarse, spec_tok, model_input_fine], dim=1)
                    patched_padding = torch.cat([pad_coarse, spec_pad, pad_fine], dim=1)
                else:
                    model_input = torch.cat([model_input_coarse, model_input_fine], dim=1)
                    patched_padding = torch.cat([pad_coarse, pad_fine], dim=1)

                # Add resolution embeddings if enabled
                if self.model.use_resolution_embeddings:
                    Ncoarse = model_input_coarse.shape[1]
                    Nfine = model_input_fine.shape[1]
                    spec_len = 1 if self.model.use_special_token else 0

                    mr_coarse = torch.zeros(Ncoarse, dtype=torch.long, device=self.device)
                    mr_spec = torch.zeros(spec_len, dtype=torch.long, device=self.device)
                    mr_fine = torch.ones(Nfine, dtype=torch.long, device=self.device)

                    mr_idx = torch.cat([mr_coarse, mr_spec, mr_fine], dim=0)
                    mr_idx = mr_idx.unsqueeze(0).expand(B, -1)

                    mr_emb = self.model.multi_resolution(mr_idx)
                    model_input = model_input + mr_emb

                # Add frequency embedding (zeros for unknown)
                freq = torch.zeros((B, 1), dtype=torch.long, device=self.device)
                f_emb = self.model.freq_emb(freq)
                model_input = model_input + f_emb

                # Forward through transformer
                model_output = self.model.stacked_transformer(model_input, patched_padding)

                # Project to output space
                output_ts = self.model.horizon_ff_layer(model_output)

                # Reshape to [B, N, H, Q]
                b, n, _ = output_ts.shape
                num_outputs = len(self.config.quantiles) + 1
                output_ts = output_ts.view(b, n, self.config.horizon_len, num_outputs)

                # Take predictions from last fine patch
                last_patch_idx = -1
                predictions = output_ts[:, last_patch_idx, :horizon_len, :]  # [B, H, Q+1]

                # Denormalize using fine context stats
                for i in range(B):
                    offset = offsets[start + i]
                    scale = scales[start + i]

                    pred = predictions[i].cpu().numpy()  # [H, Q+1]

                    # Denormalize
                    pred_denorm = pred * (scale + 1e-8) + offset

                    # Extract mean and quantiles
                    mean_forecast = pred_denorm[:, 0]
                    quantile_forecasts = pred_denorm[:, 1:]

                    result = {
                        'mean': mean_forecast,
                        'quantiles': {}
                    }

                    for q_idx, q_level in enumerate(self.config.quantiles):
                        result['quantiles'][str(q_level)] = quantile_forecasts[:, q_idx]

                    all_predictions.append(result)

        return all_predictions


if __name__ == "__main__":
    print("=" * 60)
    print("Cisco Multi-Resolution TimesFM - Standalone Inference")
    print("=" * 60)

    # Initialize model
    print("\n[1/3] Loading model...")
    model = CiscoInference(
        repo_id="cisco-ai/cisco-time-series-model-1.0-preview",
        device="auto"
    )

    # Create sample data
    print("\n[2/3] Preparing sample data...")
    np.random.seed(42)
    sample_series = np.cumsum(np.random.randn(10000)) + 100

    print(f"  - Series length: {len(sample_series)}")
    print(f"  - Series range: [{sample_series.min():.2f}, {sample_series.max():.2f}]")

    # Make forecast
    print("\n[3/3] Making forecast...")
    results = model.forecast(
        inputs=[sample_series],
        horizon_len=128,
        agg_factor=60
    )

    # Display results
    print("\n" + "=" * 60)
    print("Forecast Results")
    print("=" * 60)

    result = results[0]
    mean_forecast = result['mean']

    print(f"\nMean forecast (first 10 steps):")
    print(mean_forecast[:10])

    print(f"\nAvailable quantiles: {list(result['quantiles'].keys())}")

    print("\n✓ Inference complete!")
