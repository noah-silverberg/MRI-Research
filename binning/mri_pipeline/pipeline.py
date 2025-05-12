"""High‑level orchestration class for the MRI reconstruction workflow.

`MRIPipeline` wires together the functional sub‑modules.  Each private helper
method covers one step of the workflow so that unit‑testing individual stages
remains straightforward.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import imageio

from . import (
    data_loading as dl,
    signal_processing as sp,
    binning,
    reconstruction as recon,
    postprocessing as pp,
)


class MRIPipeline:
    """End‑to‑end reconstruction driver."""

    # ------------------------------------------------------------------#
    # Construction                                                       #
    # ------------------------------------------------------------------#
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

        # populated during runtime
        self._scan: Dict[str, Any] | None = None
        self._kspace_meas: np.ndarray | None = None
        self._row_map: np.ndarray | None = None
        self._extended_pe: int | None = None
        self._fs: float | None = None

        # peaks / cycles
        self._r_peaks: np.ndarray | None = None
        self._resp_peaks: np.ndarray | None = None
        self._resp_troughs: np.ndarray | None = None
        self._heart_rate: float | None = None

        # binning results
        self._binned_kspace: np.ndarray | None = None
        self._bin_counts: np.ndarray | None = None
        self._bin_meta: Dict[str, Any] | None = None

        # final images
        self._cine_imgs: np.ndarray | None = None

    # ------------------------------------------------------------------#
    # Public API                                                         #
    # ------------------------------------------------------------------#
    def run(self) -> None:
        """Execute the full pipeline based on the user configuration."""
        self._ingest()
        self._analyze_signals()
        self._bin()
        self._reconstruct()
        self._postprocess()
        self._export()

    # ------------------------------------------------------------------#
    # Private helpers                                                    #
    # ------------------------------------------------------------------#
    # 1. Data ingestion -------------------------------------------------#
    def _ingest(self) -> None:
        cfg_d = self.cfg["data"]
        twix_file = cfg_d["twix_file"]
        dicom_folder = cfg_d["dicom_folder"]

        scans = dl.read_twix(twix_file, include_scans=[-1])
        self._scan = scans[-1]

        raw_kspace = dl.extract_raw_frames(self._scan)
        row_map = np.where(~np.all(raw_kspace == 0, axis=(0, 2, 3)))[0]
        kspace_meas = raw_kspace[:, row_map]

        self._extended_pe = dl.get_total_phase_encodes(dicom_folder)
        self._row_map = row_map
        self._kspace_meas = kspace_meas

        # sampling frequency (TR‑1)
        method = self.cfg["processing"]["sampling_rate"]
        if method == "twix_TR":
            tr_us = self._scan["hdr"]["Phoenix"]["alTR"][0]
            self._fs = 1.0 / (tr_us * 1e-6)
        elif method == "twix_manual":
            self._fs = dl.get_frame_sampling_rate(self._scan)
        elif method == "dicom":
            # fallback to TR × phase‑encodes
            fr, _ = dl.get_frame_sampling_rate(self._scan), None
            n_meas = kspace_meas.shape[1]
            self._fs = fr * n_meas
        else:
            raise ValueError(f"Unknown sampling‑rate method '{method}'.")

    # 2. ECG / respiration analysis ------------------------------------#
    def _analyze_signals(self) -> None:
        proc = self.cfg["processing"]
        data = self.cfg["data"]
        fs = self._fs  # guaranteed set in _ingest

        # ECG -----------------------------------------------------------
        if data.get("ecg_events") is not None:
            events = dl.read_external_signal(
                data["ecg_events"], target_length=self._kspace_meas.shape[0]
            )
            self._r_peaks = np.nonzero(events > 0)[0]
        else:
            if data.get("ecg_files") is not None:
                ecg_stack = [
                    dl.read_external_signal(f, target_length=self._kspace_meas.shape[0])
                    for f in data["ecg_files"]
                ]
                ecg = np.vstack(ecg_stack).T
            else:
                cols = data["ecg_columns"]
                if cols is None:
                    raise ValueError("No ECG input specified.")
                lo, hi = map(int, cols.split(":"))
                ecg = dl.read_external_signal(
                    data["twix_file"], target_length=0
                )  # placeholder if ICE extraction added later
                ecg = ecg[:, lo:hi]
            r_list = sp.detect_r_peaks(
                ecg,
                fs,
                method=proc["ecg_method"],
                **(
                    proc.get("ecg_scipy_kwargs", {})
                    if proc["ecg_method"] == "scipy"
                    else {}
                ),
            )
            self._r_peaks = np.mean(
                np.vstack([p if p.size else np.array([0]) for p in r_list]), axis=0
            ).astype(int)

        self._heart_rate = sp.compute_heart_rate([self._r_peaks], fs)

        # Respiration ----------------------------------------------------
        if data.get("resp_events") is not None:
            resp_raw = dl.read_external_signal(
                data["resp_events"], target_length=self._kspace_meas.shape[0]
            )
        else:
            if data.get("resp_files") is not None:
                resp_raw = dl.read_external_signal(
                    data["resp_files"], target_length=self._kspace_meas.shape[0]
                )
            else:
                col = data.get("resp_columns")
                if col is None:
                    raise ValueError("No respiration input specified.")
                resp_raw = np.zeros(self._kspace_meas.shape[0])  # placeholder

        peaks, troughs = sp.detect_resp_extrema(
            resp_raw,
            fs,
            method=proc["resp_method"],
            **(
                proc.get("resp_peak_kwargs", {})
                if proc["resp_method"] == "scipy"
                else {}
            ),
        )
        self._resp_peaks = peaks
        self._resp_troughs = troughs

    # 3. Binning --------------------------------------------------------#
    def _bin(self) -> None:
        proc = self.cfg["processing"]
        if proc["resp_bin_method"] == "even":
            self._binned_kspace, self._bin_counts, self._bin_meta = binning.bin_even(
                self._kspace_meas,
                r_peaks=self._r_peaks,
                resp_peaks=self._resp_peaks,
                n_cardiac_bins=proc["num_cardiac_bins"],
                n_resp_bins=proc["num_resp_bins"],
                row_map=self._row_map,
                extended_pe=self._extended_pe,
            )
        else:
            self._binned_kspace, self._bin_counts, self._bin_meta = binning.bin_physio(
                self._kspace_meas,
                r_peaks=self._r_peaks,
                resp_peaks=self._resp_peaks,
                resp_troughs=self._resp_troughs,
                n_cardiac_bins=proc["num_cardiac_bins"],
                n_exhale_bins=proc["num_exhalation_bins"],
                n_inhale_bins=proc["num_inhalation_bins"],
                row_map=self._row_map,
                extended_pe=self._extended_pe,
            )

    # 4. Reconstruction -------------------------------------------------#
    def _reconstruct(self) -> None:
        proc = self.cfg["processing"]
        method = proc["reconstruction_method"].lower()

        c_bins, r_bins = self._binned_kspace.shape[:2]
        cine_stack: List[np.ndarray] = []

        for rb in range(r_bins):
            kspace_rb = self._binned_kspace[:, rb]
            counts_rb = self._bin_counts[:, rb]

            if method == "grappa":
                imgs = recon.grappa(
                    kspace_rb,
                    calib_region=tuple(proc["calib_region"]),
                    kernel_size=tuple(proc["kernel_size"]),
                )
            elif method == "tgrappa":
                imgs = recon.tgrappa(
                    kspace_rb,
                    calib_size=tuple(proc["calib_size"]),
                    kernel_size=tuple(proc["kernel_size"]),
                )
            else:
                imgs = recon.direct_ifft(
                    kspace_rb,
                    row_map=self._row_map,
                    extended_pe=self._extended_pe,
                    conjugate_symmetry=(method == "conj_symm"),
                    count_mask=counts_rb,
                )
            cine_stack.append(imgs)

        self._cine_imgs = np.stack(cine_stack, axis=1)  # (cardiac, resp, H, W)

    # 5. Post‑processing ------------------------------------------------#
    def _postprocess(self) -> None:
        pp_cfg = self.cfg["post_processing"]
        view = pp_cfg.get("view")
        if view == "coronal":
            self._cine_imgs = np.rot90(self._cine_imgs, k=1, axes=(2, 3))
            self._cine_imgs = np.flip(self._cine_imgs, axis=3)
            self._cine_imgs = self._cine_imgs[:, :, :, 64:-64]
        elif view == "lax":
            self._cine_imgs = np.flip(self._cine_imgs, axis=2)
            self._cine_imgs = self._cine_imgs[:, :, :, :, 64:-64]

        spec = pp_cfg.get("spec")
        if spec:
            self._cine_imgs = pp.apply_postprocessing(self._cine_imgs, spec)

    # 6. Export ---------------------------------------------------------#
    def _export(self) -> None:
        out_dir = Path(self.cfg["data"]["output_folder"])
        out_dir.mkdir(exist_ok=True)

        if self.cfg["export"]["make_gifs"]:
            duration = self.cfg["export"]["gif_duration_ms"]
            if duration is None and self._heart_rate:
                duration = 1000 * 60 / self._heart_rate / self._cine_imgs.shape[0]

            # per‑respiratory‑bin GIFs
            r_bins = self._cine_imgs.shape[1]
            for rb in range(r_bins):
                img_stack = self._cine_imgs[:, rb]
                fname = out_dir / f"cine_resp{rb}.gif"
                _save_gif(img_stack, fname, duration_ms=duration)

            # k‑space GIFs (log magnitude)
            for rb in range(r_bins):
                ks = self._binned_kspace[:, rb]
                mag = np.sqrt(np.sum(np.abs(ks) ** 2, axis=2))
                log_ks = np.log1p(mag)
                log_ks = (log_ks - log_ks.min()) / (log_ks.ptp() + 1e-9)
                _save_gif(
                    log_ks, out_dir / f"kspace_resp{rb}.gif", duration_ms=duration
                )


# -----------------------------------------------------------------------------#
# Utility: save GIF                                                            #
# -----------------------------------------------------------------------------#
def _save_gif(stack: np.ndarray, path: Path, *, duration_ms: float | int) -> None:
    """Min‑max normalise *stack* and save as a GIF."""
    stack = (stack - stack.min()) / (stack.ptp() + 1e-9)
    frames = [(f * 255).astype(np.uint8) for f in stack]
    imageio.mimsave(path, frames, duration=duration_ms / 1000)
