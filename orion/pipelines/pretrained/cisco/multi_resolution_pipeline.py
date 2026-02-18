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
Multi-resolution pipeline for TimesFM.

This demonstrates how to preprocess coarse and fine contexts before feeding
them into a multi-resolution TimesFM model (like the Cisco TSM MR).

NOTE: Standard TimesFM doesn't support multi-resolution inputs natively.
This requires a modified decoder that can handle [coarse, fine] input pairs.
"""
from typing import List, Tuple

import numpy as np
import torch


def pad_or_truncate(
    series: np.ndarray,
    target_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad (left) or truncate a time series to a target length.

    Args:
        series: 1D array of time series values.
        target_len: Desired target length after padding/truncation.

    Returns:
        Tuple of:
            - padded_series: array of shape [target_len].
            - pad_mask: array of shape [target_len], with 1.0 for padded positions
                and 0.0 for actual data.

    Example:
        >>> series = np.array([1, 2, 3])
        >>> padded, mask = pad_or_truncate(series, target_len=5)
        >>> padded
        array([0., 0., 1., 2., 3.])
        >>> mask
        array([1., 1., 0., 0., 0.])
    """
    L = len(series)

    if L == target_len:
        return series, np.zeros(target_len, dtype=np.float32)

    if L > target_len:
        # Truncate from the left (keep most recent)
        return series[-target_len:], np.zeros(target_len, dtype=np.float32)

    # Pad on the left
    pad_len = target_len - L
    padded = np.concatenate([np.zeros(pad_len, dtype=series.dtype), series])
    pad_mask = np.concatenate([
        np.ones(pad_len, dtype=np.float32),
        np.zeros(L, dtype=np.float32)
    ])

    return padded, pad_mask


def normalize_with_padding(
    context: np.ndarray,
    pad_mask: np.ndarray,
    eps: float = 1e-8,
    clamp_range: Tuple[float, float] = (-1000, 1000)
) -> Tuple[np.ndarray, float, float]:
    """Normalize context while respecting padding mask.

    Computes mean and std only over non-padded values.

    Args:
        context: Array of shape [T] containing time series values.
        pad_mask: Array of shape [T] with 1.0 for padded, 0.0 for real data.
        eps: Small epsilon for numerical stability (default: 1e-8).
        clamp_range: Tuple of (min, max) to clamp normalized values.

    Returns:
        Tuple of:
            - normalized_context: normalized array of shape [T].
            - offset: mean value used for normalization (scalar).
            - scale: std value used for normalization (scalar).

    Example:
        >>> context = np.array([0., 0., 10., 20., 30.])
        >>> pad_mask = np.array([1., 1., 0., 0., 0.])
        >>> norm, offset, scale = normalize_with_padding(context, pad_mask)
        >>> # Only values at indices 2, 3, 4 are used for stats
    """
    valid = 1.0 - pad_mask  # 1 for real data, 0 for padded
    count = max(valid.sum(), 1.0)  # prevent divide-by-zero

    # Masked mean and std
    offset = (context * valid).sum() / count
    variance = (((context - offset) * valid) ** 2).sum() / count
    scale = np.sqrt(variance)

    # Normalize
    normalized = (context - offset) / (scale + eps)

    # Zero out padded positions
    normalized = normalized * valid

    # Clamp to avoid extreme values
    normalized = np.clip(normalized, clamp_range[0], clamp_range[1])

    return normalized, float(offset), float(scale)


def preprocess_for_timesfm(
    coarse_ctx: List[float],
    fine_ctx: List[float],
    context_len_coarse: int = 512,
    context_len_fine: int = 512
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Preprocess coarse and fine contexts for TimesFM multi-resolution input.

    This performs the full preprocessing pipeline:
    1. Convert to arrays
    2. Pad/truncate to target lengths
    3. Normalize with padding awareness

    Args:
        coarse_ctx: List of coarse (long-term) context values.
        fine_ctx: List of fine (short-term) context values.
        context_len_coarse: Target length for coarse context (default: 512).
        context_len_fine: Target length for fine context (default: 512).

    Returns:
        Tuple of:
            - norm_coarse: normalized coarse context [context_len_coarse].
            - pad_mask_coarse: padding mask for coarse [context_len_coarse].
            - norm_fine: normalized fine context [context_len_fine].
            - pad_mask_fine: padding mask for fine [context_len_fine].
            - offset_fine: offset used to normalize fine context (for denorm).
            - scale_fine: scale used to normalize fine context (for denorm).

    Example:
        >>> coarse = [1.0, 2.0, 3.0]
        >>> fine = [10.0, 11.0, 12.0, 13.0]
        >>> results = preprocess_for_timesfm(coarse, fine,
        ...                                   context_len_coarse=5,
        ...                                   context_len_fine=6)
        >>> norm_coarse, pad_coarse, norm_fine, pad_fine, offset, scale = results
    """
    # Convert to arrays
    coarse_arr = np.array(coarse_ctx, dtype=np.float32)
    fine_arr = np.array(fine_ctx, dtype=np.float32)

    # Pad or truncate
    coarse_padded, mask_coarse = pad_or_truncate(coarse_arr, context_len_coarse)
    fine_padded, mask_fine = pad_or_truncate(fine_arr, context_len_fine)

    # Normalize
    norm_coarse, _, _ = normalize_with_padding(coarse_padded, mask_coarse)
    norm_fine, offset_fine, scale_fine = normalize_with_padding(fine_padded, mask_fine)

    return (
        norm_coarse,
        mask_coarse,
        norm_fine,
        mask_fine,
        offset_fine,
        scale_fine
    )


def prepare_batch_for_model(
    coarse_contexts: List[np.ndarray],
    coarse_pads: List[np.ndarray],
    fine_contexts: List[np.ndarray],
    fine_pads: List[np.ndarray],
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert preprocessed contexts into batched tensors for model input.

    Args:
        coarse_contexts: List of normalized coarse contexts, each [L_coarse].
        coarse_pads: List of coarse padding masks, each [L_coarse].
        fine_contexts: List of normalized fine contexts, each [L_fine].
        fine_pads: List of fine padding masks, each [L_fine].
        device: Device to place tensors on (default: "cpu").

    Returns:
        Tuple of:
            - batch_coarse: tensor of shape [B, L_coarse, 1].
            - batch_coarse_pad: tensor of shape [B, L_coarse, 1].
            - batch_fine: tensor of shape [B, L_fine, 1].
            - batch_fine_pad: tensor of shape [B, L_fine, 1].

    Note:
        These tensors would then be passed as:
        model.decode([batch_coarse, batch_fine],
                    [batch_coarse_pad, batch_fine_pad],
                    freq_tensor, ...)
    """
    batch_coarse = torch.as_tensor(
        np.stack(coarse_contexts),
        dtype=torch.float32
    ).unsqueeze(-1).to(device)

    batch_coarse_pad = torch.as_tensor(
        np.stack(coarse_pads),
        dtype=torch.float32
    ).unsqueeze(-1).to(device)

    batch_fine = torch.as_tensor(
        np.stack(fine_contexts),
        dtype=torch.float32
    ).unsqueeze(-1).to(device)

    batch_fine_pad = torch.as_tensor(
        np.stack(fine_pads),
        dtype=torch.float32
    ).unsqueeze(-1).to(device)

    return batch_coarse, batch_coarse_pad, batch_fine, batch_fine_pad


def full_example():
    """Complete example of the multi-resolution pipeline."""
    from context_builder import build_multi_resolution_context

    # Step 1: Create contexts from raw time series
    raw_series = np.random.randn(10000)  # Your raw fine-resolution data
    coarse_ctx, fine_ctx = build_multi_resolution_context(
        raw_series,
        agg_factor=60,  # 60 minutes -> 1 hour
        max_coarse_ctx=512,
        max_fine_ctx=512
    )

    print(f"Coarse context length: {len(coarse_ctx)}")
    print(f"Fine context length: {len(fine_ctx)}")

    # Step 2: Preprocess for model
    (norm_coarse, pad_coarse,
     norm_fine, pad_fine,
     offset, scale) = preprocess_for_timesfm(coarse_ctx, fine_ctx)

    print(f"\nNormalized coarse shape: {norm_coarse.shape}")
    print(f"Normalized fine shape: {norm_fine.shape}")
    print(f"Fine offset: {offset:.3f}, scale: {scale:.3f}")

    # Step 3: Batch multiple series (if you have more than one)
    # For single series, just wrap in a list
    coarse_batch = [norm_coarse]
    pad_coarse_batch = [pad_coarse]
    fine_batch = [norm_fine]
    pad_fine_batch = [pad_fine]

    # Step 4: Convert to tensors
    batch_coarse, batch_coarse_pad, batch_fine, batch_fine_pad = prepare_batch_for_model(
        coarse_batch,
        pad_coarse_batch,
        fine_batch,
        pad_fine_batch,
        device="cpu"
    )

    print(f"\nBatch coarse shape: {batch_coarse.shape}")  # [1, 512, 1]
    print(f"Batch fine shape: {batch_fine.shape}")  # [1, 512, 1]

    # Step 5: Feed into model (pseudocode - requires modified TimesFM decoder)
    # freq_tensor = torch.zeros((1, 1), dtype=torch.long)
    # predictions = model.decode(
    #     [batch_coarse, batch_fine],
    #     [batch_coarse_pad, batch_fine_pad],
    #     freq_tensor,
    #     horizon_len=128,
    #     agg_factor=60,
    #     offsets=[offset],
    #     scales=[scale]
    # )

    print("\n✓ Pipeline complete!")
    print("\nTo feed into TimesFM:")
    print("1. Pass coarse and fine tensors as a LIST: [batch_coarse, batch_fine]")
    print("2. Pass corresponding padding masks as a LIST: [batch_coarse_pad, batch_fine_pad]")
    print("3. Include frequency tensor (e.g., zeros for unknown frequency)")
    print("4. Store offset and scale from fine context for denormalization")


if __name__ == "__main__":
    full_example()
