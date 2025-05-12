"""Lightweight image post‑processing (rotate, flip, crop, etc.).

The caller passes a declarative *spec* list like::

    spec = [
        ("rotate", {"k": 1, "axes": (1, 2)}),
        ("flip",   {"axis": 1}),
        ("crop",   {"xmin": 64, "xmax": -64}),
    ]

Each step mutates the image stack (n_frames, H, W) and returns the result so
that sequences can be chained.
"""

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
    # placeholder
    raise NotImplementedError


@_register("flip")
def _flip(imgs: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    raise NotImplementedError


@_register("crop")
def _crop(imgs: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    raise NotImplementedError


def apply_postprocessing(
    imgs: np.ndarray, spec: List[Tuple[str, Dict[str, Any]]]
) -> np.ndarray:
    """Apply a sequence of post‑processing transformations."""
    out = imgs.copy()
    for name, params in spec:
        fn = _Registry.get(name)
        if fn is None:
            raise ValueError(f"Unknown post‑processing op '{name}'")
        out = fn(out, params)
    return out
