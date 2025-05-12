"""Configuration utilities.

This module defines the validation schema and helper functions that load and
validate the user‑provided YAML configuration. Validation is intentionally
lightweight – we raise descriptive `ValueError`s rather than relying on heavy
external libraries so that researchers can easily tweak things without needing
extra dependencies.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict

__all__ = ["load_config", "validate_config"]


def load_config(path: Path | str) -> Dict[str, Any]:
    """Load a YAML config file and immediately validate it.

    Parameters
    ----------
    path : str | Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed and validated configuration dictionary.
    """
    with Path(path).expanduser().open("r") as fh:
        cfg = yaml.safe_load(fh)
    return validate_config(cfg)


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Perform minimal sanity checks on the configuration.

    Raises
    ------
    ValueError
        If a required field is missing or contradictory options are set.
    """
    # Required top‑level sections --------
    required_sections = {"data", "processing", "post_processing"}
    missing = required_sections.difference(cfg)
    if missing:
        raise ValueError(f"Missing top‑level section(s): {', '.join(sorted(missing))}")

    # Mutually exclusive ECG/resp inputs ---
    for signal in ("ecg", "resp"):
        cols = cfg["data"].get(f"{signal}_columns")
        raw = cfg["data"].get(f"{signal}_files")
        events = cfg["data"].get(f"{signal}_events")
        provided = [v for v in (cols, raw, events) if v is not None]
        if len(provided) > 1:
            raise ValueError(
                f"For {signal.upper()} data supply *one* of columns/raw/events, got {provided}"
            )

    # Add additional checks as needed ...
    return cfg
