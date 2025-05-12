"""High‑level orchestration class.

`MRIPipeline` wires together the functional modules – it does *not* contain
heavy logic itself. All I/O and numerical work is delegated to the respective
submodules for maximal testability.
"""

from __future__ import annotations

from typing import Dict, Any, List

import numpy as np

from . import (
    data_loading as dl,
    signal_processing as sp,
    binning,
    reconstruction as recon,
    postprocessing as pp,
)


class MRIPipeline:
    """End‑to‑end reconstruction driver."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Execute the full pipeline based on the current configuration."""
        self._ingest()
        self._analyze_signals()
        self._bin()
        self._reconstruct()
        self._postprocess()
        self._export()

    # ------------------------------------------------------------------
    # Private helpers (one per major step) – implementation TBD
    # ------------------------------------------------------------------
    def _ingest(self):
        raise NotImplementedError

    def _analyze_signals(self):
        raise NotImplementedError

    def _bin(self):
        raise NotImplementedError

    def _reconstruct(self):
        raise NotImplementedError

    def _postprocess(self):
        raise NotImplementedError

    def _export(self):
        raise NotImplementedError
