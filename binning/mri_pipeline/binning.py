"""K‑space binning utilities (cardiac + respiratory).

Implements two high‑level functions:
    `bin_even()` – even respiratory partitions
    `bin_physio()` – inhale/exhale physiological model
Each returns a tuple `(binned_kspace, counts, metadata)` where metadata stores
the mapping dictionaries so post‑processing can trace provenance.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any

import numpy as np

__all__ = ["bin_even", "bin_physio"]


def _fractional_phase(idx: int, boundaries: np.ndarray) -> float:
    """Return phase (0‑1) within the current cycle defined by `boundaries`."""
    # Implementation placeholder
    raise NotImplementedError


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
    """Even respiratory partitioning."""
    raise NotImplementedError


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
    """Physiological inhale/exhale binning."""
    raise NotImplementedError
