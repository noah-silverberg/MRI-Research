"""Data‑loading helpers (TWIX, DICOM, ICE parameters).

Only *reading* happens here – no interpretation, binning, or reconstruction.
The functions below return NumPy arrays or lightweight dicts that higher‑level
modules consume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple, Dict, Any

import numpy as np
import pydicom  # lightweight; if absent, raise import error at call‑time
import twixtools

__all__ = [
    "read_twix",
    "extract_raw_frames",
    "get_total_phase_encodes",
    "get_frame_sampling_rate",
    "read_external_signal",
]


def read_twix(path: str | Path, *, include_scans: Sequence[int] | None = None) -> list:
    """Return a list of scan dicts from a .dat file (wrapper around twixtools)."""
    return twixtools.read_twix(Path(path), include_scans=include_scans)


def extract_raw_frames(scan: Dict[str, Any]) -> np.ndarray:
    """Return raw k‑space frames (n_frames, n_meas, n_coils, n_ro) from a scan.

    Intended for *undersampled* data. Zero‑filling is *not* done here.
    """
    # Implementation goes later
    raise NotImplementedError


def get_total_phase_encodes(dicom_folder: str | Path) -> int:
    """Infer the total PE lines from any DICOM slice in a folder."""
    # Implementation placeholder
    raise NotImplementedError


def get_frame_sampling_rate(scan: Dict[str, Any]) -> float:
    """Compute sampling frequency (Hz) from TWIX TimeStamp differences."""
    raise NotImplementedError


def read_external_signal(file: Path | str, *, target_length: int) -> np.ndarray:
    """Load a CSV/LOG vector and resample to `target_length`."""
    raise NotImplementedError
