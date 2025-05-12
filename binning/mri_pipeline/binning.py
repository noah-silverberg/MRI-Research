"""K‑space binning utilities (cardiac + respiratory).

Implements two high‑level functions:

    * `bin_even()`   – equal‑width respiratory partitions
    * `bin_physio()` – inhale / exhale model with separate bin counts

Both return a tuple:

    (binned_kspace, counts, metadata)

where *binned_kspace* has shape
``(n_cardiac_bins, n_resp_bins, extended_pe, n_coils, n_ro)``
(or with *n_resp_bins = n_exhale + n_inhale* for `bin_physio`),
*counts* is an int array of the same first three axes,
and *metadata* stores provenance.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any

import numpy as np


__all__ = ["bin_even", "bin_physio"]


# -----------------------------------------------------------------------------#
# Helper                                                                       #
# -----------------------------------------------------------------------------#
def _fractional_phase(idx: int, boundaries: np.ndarray) -> float:
    """Fraction (0–1) through the cycle containing *idx*.

    *boundaries* should be a **monotonic** 1‑D array of cycle start indices
    (e.g. R‑peaks).  The function returns NaN if *idx* lies outside all cycles.
    """
    pos = np.searchsorted(boundaries, idx, side="right") - 1
    if pos < 0 or pos >= len(boundaries) - 1:
        return np.nan
    start, end = boundaries[pos], boundaries[pos + 1]
    return (idx - start) / (end - start)


# -----------------------------------------------------------------------------#
# Even respiratory partitioning                                                #
# -----------------------------------------------------------------------------#
def bin_even(
    kspace_measured: np.ndarray,
    *,
    r_peaks: np.ndarray,
    resp_peaks: np.ndarray,
    n_cardiac_bins: int,
    n_resp_bins: int,
    row_map: np.ndarray,
    extended_pe: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Joint cardiac / respiratory binning with equal‑width resp bins."""
    frames, n_meas, n_coils, n_ro = kspace_measured.shape
    out_shape = (n_cardiac_bins, n_resp_bins, extended_pe, n_coils, n_ro)
    binned_sum = np.zeros(out_shape, dtype=kspace_measured.dtype)
    counts = np.zeros(out_shape[:3], dtype=np.int32)

    for f in range(frames):
        for meas_idx, phys_row in enumerate(row_map):
            global_idx = f * n_meas + meas_idx

            c_frac = _fractional_phase(global_idx, r_peaks)
            r_frac = _fractional_phase(global_idx, resp_peaks)
            if np.isnan(c_frac) or np.isnan(r_frac):
                continue

            c_bin = min(int(c_frac * n_cardiac_bins), n_cardiac_bins - 1)
            r_bin = min(int(r_frac * n_resp_bins), n_resp_bins - 1)

            binned_sum[c_bin, r_bin, phys_row] += kspace_measured[f, meas_idx]
            counts[c_bin, r_bin, phys_row] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        binned_kspace = np.where(
            counts[..., None, None] > 0,
            binned_sum / counts[..., None, None],
            0.0,
        )

    meta = {
        "row_map": row_map.copy(),
        "cardiac_bins": n_cardiac_bins,
        "resp_bins": n_resp_bins,
        "method": "even",
    }
    return binned_kspace, counts, meta


# -----------------------------------------------------------------------------#
# Physiological inhale / exhale model                                          #
# -----------------------------------------------------------------------------#
def bin_physio(
    kspace_measured: np.ndarray,
    *,
    r_peaks: np.ndarray,
    resp_peaks: np.ndarray,
    resp_troughs: np.ndarray,
    n_cardiac_bins: int,
    n_exhale_bins: int,
    n_inhale_bins: int,
    row_map: np.ndarray,
    extended_pe: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Binning that treats inhale / exhale separately."""
    total_resp_bins = n_exhale_bins + n_inhale_bins
    frames, n_meas, n_coils, n_ro = kspace_measured.shape
    out_shape = (n_cardiac_bins, total_resp_bins, extended_pe, n_coils, n_ro)
    binned_sum = np.zeros(out_shape, dtype=kspace_measured.dtype)
    counts = np.zeros(out_shape[:3], dtype=np.int32)

    # Pre‑concatenate and sort extrema for quick search
    all_ext = np.concatenate([resp_peaks, resp_troughs])
    ext_labels = np.concatenate(
        [np.ones_like(resp_peaks, dtype=bool), np.zeros_like(resp_troughs, dtype=bool)]
    )  # True = peak, False = trough
    sort_idx = np.argsort(all_ext)
    all_ext, ext_labels = all_ext[sort_idx], ext_labels[sort_idx]

    def _resp_bin(idx: int) -> int | None:
        pos = np.searchsorted(all_ext, idx, side="right") - 1
        if pos < 0 or pos >= len(all_ext) - 1:
            return None
        a_idx, b_idx = all_ext[pos], all_ext[pos + 1]
        a_is_peak = ext_labels[pos]
        # determine phase boundaries
        if a_is_peak:  # exhalation: peak -> trough
            frac = (idx - a_idx) / (b_idx - a_idx)
            bin_local = min(int(frac * n_exhale_bins), n_exhale_bins - 1)
            return bin_local  # 0‑based within exhale part
        else:  # inhalation: trough -> peak
            frac = (idx - a_idx) / (b_idx - a_idx)
            bin_local = min(int(frac * n_inhale_bins), n_inhale_bins - 1)
            return n_exhale_bins + bin_local  # offset by exhale bins

    for f in range(frames):
        for meas_idx, phys_row in enumerate(row_map):
            global_idx = f * n_meas + meas_idx

            c_frac = _fractional_phase(global_idx, r_peaks)
            rb = _resp_bin(global_idx)
            if np.isnan(c_frac) or rb is None:
                continue
            cb = min(int(c_frac * n_cardiac_bins), n_cardiac_bins - 1)

            binned_sum[cb, rb, phys_row] += kspace_measured[f, meas_idx]
            counts[cb, rb, phys_row] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        binned_kspace = np.where(
            counts[..., None, None] > 0,
            binned_sum / counts[..., None, None],
            0.0,
        )

    meta = {
        "row_map": row_map.copy(),
        "cardiac_bins": n_cardiac_bins,
        "resp_bins": {
            "exhale": n_exhale_bins,
            "inhale": n_inhale_bins,
        },
        "method": "physio",
    }
    return binned_kspace, counts, meta
