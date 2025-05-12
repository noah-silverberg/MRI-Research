"""Configuration utilities.

This module defines a default schema, merges user YAML on top, and performs
lightweight validation.  Providing only the required TWIX / DICOM paths in the
YAML is sufficient for a runnable configuration.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict


__all__ = ["load_config", "validate_config", "DEFAULT_CFG"]


# -----------------------------------------------------------------------------#
# Default configuration                                                         #
# -----------------------------------------------------------------------------#
DEFAULT_CFG: Dict[str, Any] = {
    "data": {
        # required
        "twix_file": None,
        "dicom_folder": None,
        # optional (mutually exclusive groups for each signal)
        "ecg_columns": None,
        "ecg_files": None,
        "ecg_events": None,
        "resp_columns": None,
        "resp_files": None,
        "resp_events": None,
        # output
        "output_folder": "binned_cines",
    },
    "processing": {
        # sampling‑rate inference
        "sampling_rate": "twix_TR",  # {twix_TR, twix_manual, dicom}
        # ECG
        "ecg_method": "nk",  # {nk, scipy}
        "ecg_scipy_kwargs": {"height": 0.6, "prominence": 0.2},
        # Respiration
        "resp_method": "scipy",  # {nk, scipy}
        "resp_peak_kwargs": {"height": 0.6, "prominence": 0.2},
        "resp_trough_kwargs": {"height": 0.2, "prominence": 0.15},
        # Binning
        "num_cardiac_bins": 12,
        "resp_bin_method": "even",  # {even, physio}
        "num_resp_bins": 4,
        "num_exhalation_bins": 2,
        "num_inhalation_bins": 2,
        # Reconstruction
        "reconstruction_method": "zf",  # {zf, conj_symm, grappa, tgrappa}
        "calib_region": [],
        "calib_size": [20, 20],
        "kernel_size": [5, 5],
    },
    "post_processing": {
        "view": None,  # {coronal, lax, None}
        "spec": None,  # list of (op, kwargs) tuples
    },
    "export": {
        "make_gifs": True,
        "gif_duration_ms": None,  # None → auto from HR/bin count
    },
}


# -----------------------------------------------------------------------------#
# Helpers                                                                       #
# -----------------------------------------------------------------------------#
def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *updates* into *base* (modifies *base* in place)."""
    for key, val in updates.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], val)
        else:
            base[key] = val
    return base


def load_config(path: Path | str) -> Dict[str, Any]:
    """Load a YAML config file, merge with defaults, validate, and return."""
    with Path(path).expanduser().open("r") as fh:
        user_cfg: Dict[str, Any] = yaml.safe_load(fh) or {}
    cfg = _deep_update(DEFAULT_CFG.copy(), user_cfg)
    return validate_config(cfg)


def _check_exclusive(cfg: Dict[str, Any], signal: str) -> None:
    cols = cfg["data"].get(f"{signal}_columns")
    raw = cfg["data"].get(f"{signal}_files")
    events = cfg["data"].get(f"{signal}_events")
    provided = [v for v in (cols, raw, events) if v is not None]
    if len(provided) > 1:
        raise ValueError(
            f"For {signal.upper()} data supply exactly one of columns/files/events."
        )


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Raise ValueError for missing or contradictory options."""
    if not cfg["data"]["twix_file"] or not cfg["data"]["dicom_folder"]:
        raise ValueError("Both data.twix_file and data.dicom_folder are required.")

    for sig in ("ecg", "resp"):
        _check_exclusive(cfg, sig)

    if cfg["processing"]["num_cardiac_bins"] < 1:
        raise ValueError("processing.num_cardiac_bins must be ≥ 1.")

    if cfg["processing"]["resp_bin_method"] == "even":
        if cfg["processing"]["num_resp_bins"] < 1:
            raise ValueError("processing.num_resp_bins must be ≥ 1 with even binning.")
    else:
        if (
            cfg["processing"]["num_exhalation_bins"] < 1
            or cfg["processing"]["num_inhalation_bins"] < 1
        ):
            raise ValueError(
                "processing.num_exhalation_bins and num_inhalation_bins must each be ≥ 1."
            )

    return cfg
