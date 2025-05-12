#!/usr/bin/env python3
"""
reconstruct_segmented_cine.py
--------------------------------
Create a cine from an **already‑segmented** Siemens TWIX scan.

Key points
----------
* Frame index  ← `cPhs`          (cardiac phase)
* PE‑line index ← `cLin`
* No binning / no ECG or resp processing
* Coil‑by‑coil cGRAPPA (spatial‑only) per frame
* Outputs a single GIF you can drop next to the binned cines
"""

# ── CONFIGURABLE PARAMETERS ────────────────────────────────────────────────────
TWIX_FILE = (
    "new_DATA/raw/meas_MID00058_FID00688_lax_Cine_segmented_ssfp.dat"  # path to .dat
)
OUTPUT_GIF = "segmented_cine.gif"  # file name for cine
KERNEL_SIZE = (5, 5)  # GRAPPA kernel (phase, freq)
CALIB_REGION = (42, 62)  # ()  → auto‑detect largest contiguous ACS block
VIEW = "lax"  # view angle
EXTENDED_PE_LINES = 96  # number of PE lines in the extended k‑space
GIF_FPS = 20 / 1000.0  # frames‑per‑second for output GIF
# ───────────────────────────────────────────────────────────────────────────────

import numpy as np
from tqdm import tqdm
import twixtools
import pygrappa
from utils.gif import save_images_as_gif
from utils.reconstruction import grappa_reconstruction  # uses pygrappa under the hood


def read_twix_last_scan(twix_path):
    """Return the last scan dictionary in the .dat file (usual for clinical scans)."""
    scans = twixtools.read_twix(
        twix_path,
        include_scans=[-1],
        parse_pmu=False,
        parse_geometry=False,
        verbose=False,
    )
    return scans[-1]


def build_kspace_from_mdb(scan, extended_pe_lines=None):
    """
    Assemble zero‑filled k‑space array using cPhs (frames) and cLin (phase‑encode row).
    """
    # first pass to determine matrix sizes
    max_cPhs = -1
    sample_mdb = None
    for mdb in scan["mdb"]:
        if not mdb.is_image_scan() and not mdb.is_flag_set("PATREFSCAN"):
            continue
        if mdb.cLin >= extended_pe_lines:
            # skip lines outside the extended k‑space
            continue
        max_cPhs = max(max_cPhs, mdb.cPhs)
        if sample_mdb is None:
            sample_mdb = mdb

    if sample_mdb is None:
        raise RuntimeError("No image scans found in TWIX segment.")

    n_frames = max_cPhs + 1
    n_pe = extended_pe_lines
    n_coils, n_ro = sample_mdb.data.shape

    kspace = np.zeros((n_frames, n_pe, n_coils, n_ro), dtype=complex)

    for mdb in scan["mdb"]:
        if mdb.is_image_scan() or mdb.is_flag_set("PATREFSCAN"):
            if mdb.cLin >= extended_pe_lines:
                # skip lines outside the extended k‑space
                continue
            kspace[mdb.cPhs, mdb.cLin, :, :] = mdb.data

    return kspace


def auto_calib_region(kspace_meas):
    """
    Find largest contiguous block of measured PE‑lines.  Returns (start, end) indices.
    """
    acquired = np.where(~np.all(kspace_meas == 0, axis=(0, 2, 3)))[0]
    if acquired.size == 0:
        raise RuntimeError("No measured PE‑lines detected.")
    # split into contiguous chunks, keep the longest
    blocks = np.split(acquired, np.where(np.diff(acquired) != 1)[0] + 1)
    longest = max(blocks, key=len)
    return int(longest[0]), int(longest[-1])


def reconstruct_cine(kspace, kernel_size, calib_region):
    """
    GRAPPA per frame.
    """
    if calib_region == ():
        calib_region = auto_calib_region(kspace)

    print(f"GRAPPA kernel {kernel_size}, ACS rows {calib_region[0]}:{calib_region[1]}")

    images = grappa_reconstruction(
        kspace,
        calib_region=calib_region,
        kernel_size=kernel_size,
    )

    if VIEW == "coronal":
        images = np.rot90(images, axes=(1, 2))  # rotate to axial
        images = np.flip(images, axis=2)  # inferior–superior flip
        images = images[:, 64:-64, :]  # tight FOV crop
    elif VIEW == "lax":
        images = np.flip(images, axis=1)
        images = images[:, :, 64:-64]

    return images


def main():
    print("➤ Reading TWIX …")
    scan = read_twix_last_scan(TWIX_FILE)

    print("➤ Building k‑space matrix …")
    kspace = build_kspace_from_mdb(scan, EXTENDED_PE_LINES)
    n_frames, n_pe, _, _ = kspace.shape
    print(f"   → k‑space shape: {kspace.shape}  (frames × PE × coils × RO)")

    print("➤ Reconstructing frames with cGRAPPA …")
    cine = reconstruct_cine(kspace, KERNEL_SIZE, CALIB_REGION)
    print(f"   → cine shape: {cine.shape}")

    print("➤ Saving GIF …")
    frame_duration_ms = 1000 / GIF_FPS
    save_images_as_gif(cine, OUTPUT_GIF, duration=frame_duration_ms)
    print("   Done.")


if __name__ == "__main__":
    main()
