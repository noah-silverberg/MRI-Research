"""Image reconstruction back‑ends (IFFT, GRAPPA, tGRAPPA).

All functions expect **fully‑sampled rows** in axis‑1 if `row_map` is supplied;
conjugate symmetry filling and zero‑filling are handled internally so callers
can focus on algorithm selection.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "direct_ifft",
    "grappa",
    "tgrappa",
]


def _zero_fill(
    kspace_measured: np.ndarray, *, row_map: np.ndarray, extended_pe: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (zero‑filled kspace, measured_mask)."""
    raise NotImplementedError


def direct_ifft(
    kspace_measured: np.ndarray,
    *,
    row_map: np.ndarray,
    extended_pe: int,
    conjugate_symmetry: bool = False,
    count_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Simple 2‑D IFFT with optional conjugate symmetry completion."""
    raise NotImplementedError


def grappa(
    kspace_measured: np.ndarray,
    *,
    calib_region: Tuple[int, int],
    kernel_size: Tuple[int, int] = (5, 5),
) -> np.ndarray:
    """Coil‑by‑coil GRAPPA followed by root‑sum‑of‑squares combination."""
    raise NotImplementedError


def tgrappa(
    kspace_measured: np.ndarray,
    *,
    calib_size: Tuple[int, int],
    kernel_size: Tuple[int, int] = (5, 5),
) -> np.ndarray:
    """Temporal GRAPPA across cine frames."""
    raise NotImplementedError
