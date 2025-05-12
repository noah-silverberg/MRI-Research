"""Data‑loading helpers (TWIX, DICOM, ICE parameters).

Only *reading* happens here – no interpretation, binning, or reconstruction.
The functions below return NumPy arrays or lightweight dicts for higher‑level
modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple, Dict, Any

import numpy as np
import pydicom
import twixtools
from scipy import signal


__all__ = [
    "read_twix",
    "extract_raw_frames",
    "get_total_phase_encodes",
    "get_frame_sampling_rate",
    "read_external_signal",
]


# -----------------------------------------------------------------------------#
# TWIX                                                                          #
# -----------------------------------------------------------------------------#
def read_twix(path: str | Path, *, include_scans: Sequence[int] | None = None) -> list:
    """Read a Siemens .dat file and return a list of scan dictionaries."""
    return twixtools.read_twix(Path(path).expanduser(), include_scans=include_scans)


def extract_raw_frames(scan: Dict[str, Any]) -> np.ndarray:
    """Stack raw *image_scan* packets into (n_frames, n_meas, n_coils, n_ro)."""
    image_pkts = [m for m in scan["mdb"] if m.is_image_scan()]
    if not image_pkts:
        raise ValueError("Scan contains no image_scan packets.")

    n_frames = max(m.cRep for m in image_pkts) + 1
    n_meas = max(m.cLin for m in image_pkts) + 1
    n_coils, n_ro = image_pkts[0].data.shape

    out = np.zeros((n_frames, n_meas, n_coils, n_ro), dtype=np.complex64)
    for m in image_pkts:
        out[m.cRep, m.cLin] = m.data
    return out


# -----------------------------------------------------------------------------#
# DICOM                                                                         #
# -----------------------------------------------------------------------------#
def _first_dicom(folder: str | Path):
    folder = Path(folder)
    for f in folder.iterdir():
        if f.suffix.lower() == ".dcm":
            return pydicom.dcmread(f)
    raise FileNotFoundError(f"No DICOM files found in {folder}")


def get_total_phase_encodes(dicom_folder: str | Path) -> int:
    """Return the PE dimension from AcquisitionMatrix or Rows."""
    ds = _first_dicom(dicom_folder)
    if hasattr(ds, "AcquisitionMatrix") and ds.AcquisitionMatrix[1] > 0:
        return int(ds.AcquisitionMatrix[1])
    if hasattr(ds, "Rows"):
        return int(ds.Rows)
    raise ValueError("Unable to determine total phase‑encodes from DICOM metadata.")


def get_frame_sampling_rate(scan: Dict[str, Any]) -> float:
    """Compute temporal sampling rate (Hz) from TWIX TimeStamp ticks."""
    ticks = [m.mdh.TimeStamp for m in scan["mdb"] if m.is_image_scan()]
    if len(ticks) < 2:
        raise ValueError("Insufficient TimeStamp entries to estimate TR.")
    dt_ticks = np.diff(sorted(ticks))
    dt_sec = np.median(dt_ticks) * 2.5e-6  # 2.5 µs resolution
    if dt_sec <= 0:
        raise ValueError("Non‑positive TimeStamp delta encountered.")
    return 1.0 / dt_sec


# -----------------------------------------------------------------------------#
# External CSV / LOG signals                                                    #
# -----------------------------------------------------------------------------#
def read_external_signal(file: Path | str, *, target_length: int) -> np.ndarray:
    """Load second‑column values from *file* and resample to *target_length*."""
    vec = np.loadtxt(file, usecols=1)
    if vec.size == 0:
        raise ValueError(f"No numeric data found in {file}")
    return signal.resample(vec, target_length).astype(np.float32)
