"""ECG and respiratory signal utilities.

This keeps NeuroKit2‑based and SciPy‑based peak detectors in one place so that
the rest of the pipeline only needs *indices* and *fractional phase* arrays.
"""

from __future__ import annotations

from typing import List, Tuple, Sequence

import numpy as np
import scipy.signal as signal
import neurokit2 as nk

__all__ = [
    "detect_r_peaks",
    "compute_heart_rate",
    "detect_resp_extrema",
]


def detect_r_peaks(
    ecg: np.ndarray, fs: float, *, method: str = "nk"
) -> List[np.ndarray]:
    """Return R‑peak indices for each ECG channel.

    Parameters
    ----------
    ecg : ndarray (n_samples, n_channels)
        Raw (or pre‑filtered) ECG.
    fs : float
        Sampling frequency [Hz].
    method : {"nk", "scipy"}
        Detection backend.
    """
    raise NotImplementedError


def compute_heart_rate(r_peaks: Sequence[np.ndarray], fs: float) -> float:
    """Average heart rate (BPM) from multi‑channel R‑peaks."""
    raise NotImplementedError


def detect_resp_extrema(
    resp: np.ndarray,
    fs: float,
    *,
    method: str = "scipy",
    height: float | None = None,
    prominence: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (peaks, troughs) indices for a single‑channel respiratory trace."""
    raise NotImplementedError
