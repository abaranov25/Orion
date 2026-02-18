"""
Simple context builder for TimesFM multi-resolution forecasting.
Based on the repository: https://github.com/splunk/cisco-time-series-model/tree/main

This module extracts the essential context-building logic for creating
long-term (coarse) and short-term (fine) resolution contexts from time series data.
"""
import numpy as np


def build_coarse_context(series: np.ndarray, max_coarse_ctx: int = 512, block: int = 60):
    """
    Construct coarse (long-term) context by aggregating fine samples.

    Takes the rightmost (max_coarse_ctx * block) raw samples, partitions them
    into consecutive non-overlapping blocks, and computes the mean of each block.
    """
    needed_raw = max_coarse_ctx * block
    raw_slice = series[-needed_raw:]

    remainder = len(raw_slice) % block
    if remainder != 0:
        raw_slice = raw_slice[remainder:]

    coarse = []
    for i in range(0, len(raw_slice), block):
        block_vals = raw_slice[i:i + block]
        if len(block_vals) < block:
            break
        coarse.append(float(np.mean(block_vals)))

    return coarse[-max_coarse_ctx:]


def build_fine_context(series: np.ndarray, fine_len: int = 512):
    """
    Extract fine (short-term) context from the rightmost samples.
    """
    if isinstance(series, np.ndarray):
        series = series.tolist()
    return series[-fine_len:]


def build_multi_resolution_context(
        series: np.ndarray, agg_factor: int = 60, max_coarse_ctx: int = 512, max_fine_ctx: int = 512):
    """
    Build both coarse and fine resolution contexts from a time series.
    This is the main function for creating multi-resolution inputs for TimesFM.
    """
    coarse_ctx = build_coarse_context(
        series,
        max_coarse_ctx=max_coarse_ctx,
        block=agg_factor
    )
    fine_ctx = build_fine_context(series, fine_len=max_fine_ctx)

    return coarse_ctx, fine_ctx


if __name__ == "__main__":
    series = np.arange(60000)
    coarse, fine = build_multi_resolution_context(
        series, agg_factor=60, max_coarse_ctx=30, max_fine_ctx=30)
    print(coarse)
    print(fine)
