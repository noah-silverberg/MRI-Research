"""ECG and respiratory signal utilities.

NeuroKit2‑ and SciPy‑based detectors are wrapped here so downstream code deals
only with index arrays and derived metrics.
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


# -----------------------------------------------------------------------------#
# ECG R‑peak detection                                                          #
# -----------------------------------------------------------------------------#
def detect_r_peaks(
    ecg: np.ndarray,
    fs: float,
    *,
    method: str = "nk",
    height: float | None = None,
    prominence: float | None = None,
) -> List[np.ndarray]:
    """Detect R‑peaks for each channel.

    Parameters
    ----------
    ecg
        ECG array of shape (n_samples, n_channels) or (n_samples,).
    fs
        Sampling frequency [Hz].
    method
        'nk' (NeuroKit2, default) or 'scipy'.
    height, prominence
        Thresholds for SciPy backend.

    Returns
    -------
    list
        List of integer index arrays, one per channel.
    """
    ecg = np.asarray(ecg)
    if ecg.ndim == 1:
        ecg = ecg[:, None]

    peaks: List[np.ndarray] = []

    if method.lower() == "nk":
        for ch in ecg.T:
            clean = nk.ecg_clean(ch, sampling_rate=fs)
            _, info = nk.ecg_peaks(clean, sampling_rate=fs)
            peaks.append(np.asarray(info["ECG_R_Peaks"], dtype=int))
        return peaks

    if method.lower() == "scipy":
        # Simple high‑pass filter (~5 Hz) then find_peaks
        b, a = signal.butter(2, 5 / (fs / 2), btype="highpass")
        for ch in ecg.T:
            filt = signal.filtfilt(b, a, ch)
            idx, _ = signal.find_peaks(
                filt,
                height=height,
                prominence=prominence,
                distance=int(0.25 * fs),  # cap at 240 bpm
            )
            peaks.append(idx.astype(int))
        return peaks

    raise ValueError(f"Unknown ECG detection method '{method}'.")


# -----------------------------------------------------------------------------#
# Heart‑rate helper                                                             #
# -----------------------------------------------------------------------------#
def compute_heart_rate(r_peaks: Sequence[np.ndarray], fs: float) -> float:
    """Return average BPM across all channels."""
    rr_all = [np.diff(ch) for ch in r_peaks if len(ch) > 1]
    if not rr_all:
        return float("nan")
    rr = np.concatenate(rr_all)
    return 60.0 * fs / np.mean(rr)


# -----------------------------------------------------------------------------#
# Respiratory peak / trough detection                                           #
# -----------------------------------------------------------------------------#
def detect_resp_extrema(
    resp: np.ndarray,
    fs: float,
    *,
    method: str = "scipy",
    height: float | None = None,
    prominence: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (peaks, troughs) for a single‑channel respiratory trace."""
    resp = np.asarray(resp).squeeze()

    if method.lower() == "nk":
        _, info = nk.rsp_peaks(resp, sampling_rate=fs)
        return (
            np.asarray(info["RSP_Peaks"], dtype=int),
            np.asarray(info["RSP_Troughs"], dtype=int),
        )

    if method.lower() == "scipy":
        norm = (resp - resp.min()) / (resp.ptp() + 1e-9)
        peaks, _ = signal.find_peaks(norm, height=height, prominence=prominence)
        troughs, _ = signal.find_peaks(-norm, height=height, prominence=prominence)
        return peaks.astype(int), troughs.astype(int)

    raise ValueError(f"Unknown respiratory detection method '{method}'.")
