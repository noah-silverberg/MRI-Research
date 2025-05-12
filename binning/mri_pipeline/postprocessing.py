"""Lightweight image post‑processing (rotate, flip, crop, etc.)."""

from __future__ import annotations

from typing import List, Tuple, Dict, Callable, Any

import numpy as np


__all__ = ["apply_postprocessing"]


_Registry: Dict[str, Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = {}


def _register(name: str):
    def decorator(fn):
        _Registry[name] = fn
        return fn

    return decorator


@_register("rotate")
def _rotate(imgs: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Rotate `imgs` by `k*90°` about the given axes."""
    k = params.get("k", 1)
    axes = params.get("axes", (1, 2))
    return np.rot90(imgs, k=k, axes=axes)


@_register("flip")
def _flip(imgs: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Flip along `axis`."""
    axis = params.get("axis", 1)
    return np.flip(imgs, axis=axis)


@_register("crop")
def _crop(imgs: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Crop using [xmin, xmax) and/or [ymin, ymax) (supports negative indices)."""
    xmin = params.get("xmin", None)
    xmax = params.get("xmax", None)
    ymin = params.get("ymin", None)
    ymax = params.get("ymax", None)

    xslice = slice(xmin, xmax)
    yslice = slice(ymin, ymax)
    return imgs[:, yslice, xslice]


def apply_postprocessing(
    imgs: np.ndarray, spec: List[Tuple[str, Dict[str, Any]]]
) -> np.ndarray:
    """Apply a sequence of post‑processing transformations."""
    out = imgs.copy()
    if spec is None:
        return out
    for name, params in spec:
        fn = _Registry.get(name)
        if fn is None:
            raise ValueError(f"Unknown post‑processing op '{name}'.")
        out = fn(out, params or {})
    return out
