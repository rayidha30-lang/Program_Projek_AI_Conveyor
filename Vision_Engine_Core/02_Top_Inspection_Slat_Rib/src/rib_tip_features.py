"""Shared 1D contour feature extraction and pure-NumPy logistic inference for Jalur 1."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "pv",
    "sag",
    "idev",
    "tdev",
    "left_max",
    "right_max",
    "asym",
    "left_slope",
    "right_slope",
    "coh_run",
)

# Canonical length matching ProfileJson (~85 points when sampled every 4th contour point)
_CANONICAL_PROFILE_LEN = 85


def _resample_profile(pts: np.ndarray, target_len: int = _CANONICAL_PROFILE_LEN) -> np.ndarray:
    """Resample 1D contour array to canonical length so training and runtime match."""
    if len(pts) == target_len:
        return pts
    src_x = np.linspace(0.0, 1.0, len(pts), dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.interp(dst_x, src_x, pts).astype(np.float32)


def extract_tip_classifier_features(
    contour_ys: Sequence[float] | np.ndarray | None,
    pv: float,
    sag: float,
    idev: float,
    tdev: float,
) -> np.ndarray | None:
    """Extract 10 morphology + metrology features from a 1D rib contour."""
    if contour_ys is None:
        return None
    pts = np.asarray(contour_ys, dtype=np.float32).ravel()
    if len(pts) < 20 or not np.all(np.isfinite(pts)):
        return None

    pts = _resample_profile(pts, _CANONICAL_PROFILE_LEN)
    n = len(pts)
    xs = np.arange(n, dtype=np.float32)

    mid_s = max(1, int(n * 0.2))
    mid_e = min(n - 1, int(n * 0.8))
    if mid_e - mid_s < 4:
        return None

    m, c = np.polyfit(xs[mid_s:mid_e], pts[mid_s:mid_e], 1)
    dev = pts - (m * xs + c)

    # Skip ~2.5% outermost bevel/chamfer points to avoid corner noise & angle shift false NGs
    trim = max(1, int(n * 0.025))
    tip_end = max(trim + 5, int(n * 0.13))
    left_dev = dev[trim:tip_end]
    right_dev = dev[-tip_end:-trim]

    left_slope = float((left_dev[0] - left_dev[-1]) / float(len(left_dev)))
    right_slope = float((right_dev[-1] - right_dev[0]) / float(len(right_dev)))
    left_max = float(np.percentile(np.abs(left_dev), 85))
    right_max = float(np.percentile(np.abs(right_dev), 85))
    asym = abs(left_max - right_max)

    left_coh = max(int(np.sum(left_dev > 3.0)), int(np.sum(left_dev < -3.0)))
    right_coh = max(int(np.sum(right_dev > 3.0)), int(np.sum(right_dev < -3.0)))
    coh_run = float(max(left_coh, right_coh))

    return np.array(
        [
            float(pv),
            float(sag),
            float(idev),
            float(tdev),
            left_max,
            right_max,
            asym,
            left_slope,
            right_slope,
            coh_run,
        ],
        dtype=np.float32,
    )


def predict_tip_ng_prob(
    features: Sequence[float] | np.ndarray,
    model: Mapping[str, Any],
) -> float:
    """Compute P(NG) using standardized pure-NumPy logistic regression."""
    x = np.asarray(features, dtype=np.float64)
    mean = np.asarray(model["scaler_mean"], dtype=np.float64)
    scale = np.asarray(model["scaler_scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    bias = float(model["bias"])

    safe_scale = np.where(np.abs(scale) < 1e-9, 1.0, scale)
    z = float(np.dot((x - mean) / safe_scale, weights) + bias)
    if z >= 0.0:
        return float(1.0 / (1.0 + math.exp(-z)))
    ez = math.exp(z)
    return float(ez / (1.0 + ez))


def load_tip_classifier_model(model_path: str | Path) -> dict[str, Any] | None:
    """Load and validate a serialized JSON tip classifier model; return None on failure."""
    path = Path(model_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        required = ("feature_names", "scaler_mean", "scaler_scale", "weights", "bias", "prob_threshold")
        if not all(k in data for k in required):
            return None
        n_feat = len(FEATURE_NAMES)
        if (
            tuple(data["feature_names"]) != FEATURE_NAMES
            or len(data["scaler_mean"]) != n_feat
            or len(data["scaler_scale"]) != n_feat
            or len(data["weights"]) != n_feat
        ):
            return None
        return data
    except Exception:
        return None
