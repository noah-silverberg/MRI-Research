"""Image‑reconstruction back‑ends.

If *row_map* is supplied, `kspace_measured` is expected as
``(n_frames, n_meas, n_coils, n_ro)`` and will be zero‑filled into an
``extended_pe`` dimension.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import pygrappa
except ImportError:
    pygrappa = None  # pygrappa is optional


__all__ = ["direct_ifft", "grappa", "tgrappa"]


# -----------------------------------------------------------------------------#
# Helpers                                                                       #
# -----------------------------------------------------------------------------#
def _zero_fill(
    kspace_measured: np.ndarray,
    *,
    row_map: np.ndarray,
    extended_pe: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return zero‑filled k‑space and a boolean measured mask."""
    frames, n_meas, n_coils, n_ro = kspace_measured.shape
    out = np.zeros((frames, extended_pe, n_coils, n_ro), dtype=kspace_measured.dtype)
    out[:, row_map] = kspace_measured
    mask = np.zeros((frames, extended_pe), dtype=bool)
    mask[:, row_map] = True
    return out, mask


def _fill_conjugate_symmetry(kspace: np.ndarray, mask: np.ndarray) -> None:
    """In‑place fill of missing rows using k_y ↔ −k_y conjugate."""
    frames, rows, _, _ = kspace.shape
    center = rows // 2
    for f in range(frames):
        for r in range(rows):
            if mask[f, r]:
                continue
            sym = (2 * center - r) % rows
            if mask[f, sym]:
                kspace[f, r] = np.conj(kspace[f, sym])


# -----------------------------------------------------------------------------#
# Direct IFFT                                                                   #
# -----------------------------------------------------------------------------#
def direct_ifft(
    kspace_measured: np.ndarray,
    *,
    row_map: np.ndarray,
    extended_pe: int,
    conjugate_symmetry: bool = False,
    count_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Root‑sum‑of‑squares reconstruction of each frame."""
    kspace, measured_mask = _zero_fill(
        kspace_measured,
        row_map=row_map,
        extended_pe=extended_pe,
    )
    if conjugate_symmetry:
        _fill_conjugate_symmetry(kspace, measured_mask)

    if count_mask is not None:
        # down‑weight rows with fewer samples (simple average)
        kspace = np.where(
            count_mask[:, :, None, None] > 0,
            kspace / np.clip(count_mask[:, :, None, None], 1, None),
            kspace,
        )

    img_coils = np.fft.ifft2(kspace, axes=(1, 3))
    img_mag = np.sqrt(np.sum(np.abs(img_coils) ** 2, axis=2))
    return np.fft.fftshift(img_mag, axes=(1, 2))


# -----------------------------------------------------------------------------#
# GRAPPA                                                                        #
# -----------------------------------------------------------------------------#
def grappa(
    kspace_measured: np.ndarray,
    *,
    calib_region: Tuple[int, int],
    kernel_size: Tuple[int, int] = (5, 5),
) -> np.ndarray:
    """Per‑frame GRAPPA reconstruction (requires *pygrappa*)."""
    if pygrappa is None:
        raise ImportError("pygrappa not installed; cannot run GRAPPA reconstruction.")

    frames, _, n_coils, _ = kspace_measured.shape
    out_imgs = []
    for f in range(frames):
        ks = kspace_measured[f]
        start, end = calib_region
        calib = ks[start : end + 1]  # inclusive slice
        recon = pygrappa.cgrappa(ks, calib, kernel_size=kernel_size, coil_axis=1)
        img = np.sqrt(np.sum(np.abs(np.fft.ifft2(recon)) ** 2, axis=1))
        out_imgs.append(np.fft.fftshift(img))
    return np.stack(out_imgs)


# -----------------------------------------------------------------------------#
# Temporal GRAPPA                                                               #
# -----------------------------------------------------------------------------#
def tgrappa(
    kspace_measured: np.ndarray,
    *,
    calib_size: Tuple[int, int],
    kernel_size: Tuple[int, int] = (5, 5),
) -> np.ndarray:
    """Temporal GRAPPA across cine frames (requires *pygrappa*)."""
    if pygrappa is None:
        raise ImportError("pygrappa not installed; cannot run tGRAPPA reconstruction.")

    # rearrange to (phase, freq, coils, time)
    ks = np.transpose(kspace_measured, (1, 3, 2, 0))
    recon = pygrappa.tgrappa(
        ks,
        calib_size=calib_size,
        kernel_size=kernel_size,
        coil_axis=-2,
        time_axis=-1,
    )
    recon = np.transpose(recon, (3, 0, 2, 1))  # back to (t, phase, coils, freq)
    img = np.sqrt(np.sum(np.abs(np.fft.ifft2(recon)) ** 2, axis=2))
    return np.fft.fftshift(img, axes=(1, 2))
