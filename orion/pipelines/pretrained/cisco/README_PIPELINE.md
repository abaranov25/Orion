# Multi-Resolution TimesFM Pipeline Guide

This guide explains how to feed coarse (long-term) and fine (short-term) contexts into a multi-resolution TimesFM model.

## Overview

The pipeline consists of 5 main steps:

```
Raw Time Series → Context Building → Padding/Truncation → Normalization → Batching → Model Input
```

## Step-by-Step Process

### 1. Build Multi-Resolution Contexts

**Input:** Raw time series (e.g., 10,000 minute-level data points)

**Output:** Coarse and fine contexts

```python
from context_builder import build_multi_resolution_context

raw_series = np.array([...])  # Your raw data

coarse_ctx, fine_ctx = build_multi_resolution_context(
    raw_series,
    agg_factor=60,       # 60 minutes → 1 hour
    max_coarse_ctx=512,  # Keep last 512 hours
    max_fine_ctx=512     # Keep last 512 minutes
)

# coarse_ctx: List[float] of length ≤ 512 (hourly aggregates)
# fine_ctx: List[float] of length ≤ 512 (minute-level data)
```

**What happens:**
- **Coarse context:** Takes rightmost 512×60 = 30,720 points, aggregates into 512 hourly means
- **Fine context:** Takes rightmost 512 points as-is

---

### 2. Pad or Truncate to Fixed Length

**Input:** Variable-length contexts

**Output:** Fixed-length contexts with padding masks

```python
from multi_resolution_pipeline import pad_or_truncate

# Pad both contexts to exactly 512 length
coarse_padded, mask_coarse = pad_or_truncate(np.array(coarse_ctx), 512)
fine_padded, mask_fine = pad_or_truncate(np.array(fine_ctx), 512)

# Shapes: [512], [512]
# Padding is added on the LEFT (oldest timestamps)
# Masks: 1.0 = padded, 0.0 = real data
```

**Why:** TimesFM expects fixed-length inputs for batching.

---

### 3. Normalize with Padding Awareness

**Input:** Padded contexts with masks

**Output:** Normalized contexts (mean=0, std=1) with normalization stats

```python
from multi_resolution_pipeline import normalize_with_padding

# Normalize (stats computed only on non-padded values)
norm_coarse, _, _ = normalize_with_padding(coarse_padded, mask_coarse)
norm_fine, offset_fine, scale_fine = normalize_with_padding(fine_padded, mask_fine)

# norm_coarse, norm_fine: [512] normalized arrays
# offset_fine, scale_fine: scalars for denormalizing predictions later
```

**Key:** Only the fine context's offset/scale are saved for denormalization!

---

### 4. Batch for Model Input

**Input:** Normalized contexts (possibly multiple series)

**Output:** PyTorch tensors ready for model

```python
from multi_resolution_pipeline import prepare_batch_for_model

# If you have multiple series, collect them in lists:
coarse_batch = [norm_coarse_series1, norm_coarse_series2, ...]
pad_coarse_batch = [mask_coarse_1, mask_coarse_2, ...]
fine_batch = [norm_fine_series1, norm_fine_series2, ...]
pad_fine_batch = [mask_fine_1, mask_fine_2, ...]

# Convert to tensors
batch_coarse, batch_coarse_pad, batch_fine, batch_fine_pad = prepare_batch_for_model(
    coarse_batch,
    pad_coarse_batch,
    fine_batch,
    pad_fine_batch,
    device="cuda:0"  # or "cpu"
)

# Shapes: [B, 512, 1] where B = batch size
```

---

### 5. Feed into Multi-Resolution TimesFM

**Input:** Batched tensors

**Output:** Forecasts

```python
# Note: This requires a MODIFIED TimesFM decoder (like Cisco's PatchedTSMultiResolutionDecoder)
# Standard TimesFM does NOT support multi-resolution inputs!

freq_tensor = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

predictions = model.decode(
    input_ts=[batch_coarse, batch_fine],           # LIST of 2 tensors
    paddings=[batch_coarse_pad, batch_fine_pad],   # LIST of 2 padding tensors
    freq=freq_tensor,
    horizon_len=128,                                # How many steps to forecast
    agg_factor=60,                                  # Same as context building
    offsets=[offset_fine_1, offset_fine_2, ...],   # List of offsets (one per series)
    scales=[scale_fine_1, scale_fine_2, ...],       # List of scales (one per series)
    output_patch_len=128                            # Steps per decode iteration
)

# predictions: List of dicts with 'mean' and 'quantiles' keys
```

---

## Key Format Details

### Model Input Format

The multi-resolution decoder expects **lists** not tensors:

```python
# ✓ CORRECT:
input_ts = [coarse_tensor, fine_tensor]
paddings = [coarse_pad_tensor, fine_pad_tensor]

# ✗ WRONG:
input_ts = torch.cat([coarse_tensor, fine_tensor], dim=1)  # Don't concatenate!
```

### Tensor Shapes

```
batch_coarse:     [B, 512, 1]  # B series, 512 coarse steps, 1 channel
batch_coarse_pad: [B, 512, 1]  # Padding mask for coarse
batch_fine:       [B, 512, 1]  # B series, 512 fine steps, 1 channel
batch_fine_pad:   [B, 512, 1]  # Padding mask for fine
freq:             [B, 1]       # Frequency index per series (often zeros)
```

### Why 3D (B, T, 1)?

The last dimension is for multi-channel support (e.g., multiple features). For univariate forecasting, it's always 1.

---

## Complete Working Example

```python
from context_builder import build_multi_resolution_context
from multi_resolution_pipeline import preprocess_for_timesfm, prepare_batch_for_model
import numpy as np
import torch

# 1. Raw data (e.g., 10,000 minutes of observations)
raw_series = np.random.randn(10000)

# 2. Build contexts
coarse_ctx, fine_ctx = build_multi_resolution_context(
    raw_series, agg_factor=60
)

# 3. Preprocess
(norm_coarse, pad_coarse, norm_fine, pad_fine, 
 offset, scale) = preprocess_for_timesfm(coarse_ctx, fine_ctx)

# 4. Batch (wrap single series in lists)
batch_coarse, batch_coarse_pad, batch_fine, batch_fine_pad = prepare_batch_for_model(
    [norm_coarse], [pad_coarse], [norm_fine], [pad_fine]
)

# 5. Create frequency tensor
freq = torch.zeros((1, 1), dtype=torch.long)

# 6. Forecast (requires modified decoder)
# predictions = model.decode(
#     [batch_coarse, batch_fine],
#     [batch_coarse_pad, batch_fine_pad],
#     freq,
#     horizon_len=128,
#     agg_factor=60,
#     offsets=[offset],
#     scales=[scale]
# )
```

---

## Important Notes

### 1. Standard TimesFM Doesn't Support This

The standard TimesFM model from Google expects a **single tensor** as input, not multi-resolution [coarse, fine] pairs. The Cisco vendored code adds:

- `PatchedTSMultiResolutionDecoder` (modified decoder)
- `CiscoTsmMR` (wrapper that handles multi-resolution)

### 2. Denormalization

The model outputs normalized predictions. To convert back to original scale:

```python
# Model returns normalized predictions
prediction_normalized = predictions[0]['mean']  # array of forecasted values

# Denormalize
prediction_original = prediction_normalized * scale + offset
```

The `decode` method in the Cisco code does this automatically using the provided `offsets` and `scales`.

### 3. Autoregressive Decoding

The `decode` method runs **autoregressively**:
1. Predicts next 128 fine steps (output_patch_len)
2. Appends predictions to fine context
3. Aggregates predictions into coarse context
4. Repeats until full horizon is forecasted

This is why `agg_factor` must match between context building and decoding!

---

## Summary

**Pipeline:**
```
Raw Data
  ↓
[build_multi_resolution_context]
  ↓
Coarse (512 hrs) + Fine (512 mins)
  ↓
[pad_or_truncate]
  ↓
Fixed-length contexts + masks
  ↓
[normalize_with_padding]
  ↓
Normalized contexts + stats
  ↓
[prepare_batch_for_model]
  ↓
PyTorch tensors [B, 512, 1]
  ↓
[model.decode with LISTS]
  ↓
Predictions (auto-denormalized)
```

**Key Insight:** The model processes **both** long-term patterns (coarse/hourly) and short-term patterns (fine/minute-level) simultaneously by concatenating them internally with special embeddings to distinguish resolution levels.
