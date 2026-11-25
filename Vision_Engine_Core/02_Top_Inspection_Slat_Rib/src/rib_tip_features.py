"""Shared 1D contour feature extraction, ONNX/NumPy 1D-CNN, and Mahalanobis inference."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import onnxruntime as _ort
    HAS_ORT = True
except ImportError:
    _ort = None
    HAS_ORT = False

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


def extract_contour_1d_signal(
    contour_ys: Sequence[float] | np.ndarray | None,
    res_scale: float = 1.0,
) -> np.ndarray | None:
    """Extract normalized (2, 85) 1D signal: [residual_dy, grad_dy] from center-body regression."""
    if contour_ys is None:
        return None
    pts = np.asarray(contour_ys, dtype=np.float32).ravel()
    if len(pts) < 20 or not np.all(np.isfinite(pts)):
        return None

    pts_85 = _resample_profile(pts, _CANONICAL_PROFILE_LEN)
    xs = np.arange(_CANONICAL_PROFILE_LEN, dtype=np.float32)
    mid_s = max(1, int(_CANONICAL_PROFILE_LEN * 0.2))
    mid_e = min(_CANONICAL_PROFILE_LEN - 1, int(_CANONICAL_PROFILE_LEN * 0.8))
    if mid_e - mid_s < 4:
        return None

    m, c = np.polyfit(xs[mid_s:mid_e], pts_85[mid_s:mid_e], 1)
    res_1d = (pts_85 - (m * xs + c)).astype(np.float32)
    if res_scale is not None and abs(float(res_scale) - 1.0) > 1e-4:
        res_1d = (res_1d * float(res_scale)).astype(np.float32)
    grad_1d = np.gradient(res_1d).astype(np.float32)
    return np.stack([res_1d, grad_1d], axis=0)


def extract_tip_classifier_features(
    contour_ys: Sequence[float] | np.ndarray | None,
    pv: float,
    sag: float,
    idev: float,
    tdev: float,
    res_scale: float = 1.0,
) -> np.ndarray | None:
    """Extract 10 morphology + metrology features from a 1D rib contour."""
    sig = extract_contour_1d_signal(contour_ys, res_scale=res_scale)
    if sig is None:
        return None

    dev = sig[0]
    n = len(dev)

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


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return float(1.0 / (1.0 + math.exp(-z)))
    ez = math.exp(z)
    return float(ez / (1.0 + ez))


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
    return _sigmoid(z)


def compute_mahalanobis_distance(
    features: Sequence[float] | np.ndarray,
    mahal_cfg: Mapping[str, Any],
) -> float:
    """Compute regularized Mahalanobis distance on standardized 10 morphology features."""
    x = np.asarray(features, dtype=np.float64)
    s_mean = np.asarray(mahal_cfg["scaler_mean"], dtype=np.float64)
    s_scale = np.asarray(mahal_cfg["scaler_scale"], dtype=np.float64)
    safe_scale = np.where(np.abs(s_scale) < 1e-9, 1.0, s_scale)
    xs = (x - s_mean) / safe_scale

    loc = np.asarray(mahal_cfg["location"], dtype=np.float64)
    prec = np.asarray(mahal_cfg["precision"], dtype=np.float64)
    diff = xs - loc
    d2 = float(np.dot(diff @ prec, diff))
    return float(math.sqrt(max(0.0, d2)))


def _conv1d_same_relu(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, apply_relu: bool = True) -> np.ndarray:
    """Pure-NumPy 1D convolution with padding=kernel//2 matching PyTorch Conv1d."""
    out_c, in_c, k = weight.shape
    pad = k // 2
    x_pad = np.pad(x, ((0, 0), (pad, pad)), mode="constant")
    # Sliding windows: (in_c, L_out, k) -> dot with weight (out_c, in_c, k)
    windows = np.lib.stride_tricks.sliding_window_view(x_pad, k, axis=1)
    y = np.einsum("ock,clk->ol", weight, windows, optimize=True) + bias[:, None]
    if apply_relu:
        np.maximum(y, 0.0, out=y)
    return y.astype(np.float32)


def _predict_1dcnn_numpy(sig_norm: np.ndarray, np_weights: Mapping[str, Any]) -> float:
    """Run folded 3-layer Contour1DCNN inference in pure NumPy (zero runtime dependencies)."""
    h = sig_norm
    for layer in np_weights["conv_layers"]:
        w = np.asarray(layer["weight"], dtype=np.float32)
        b = np.asarray(layer["bias"], dtype=np.float32)
        h = _conv1d_same_relu(h, w, b, apply_relu=True)
    pooled = np.concatenate([h.mean(axis=1), h.max(axis=1)], axis=0)
    fc_w = np.asarray(np_weights["fc_weight"], dtype=np.float32)
    fc_b = float(np_weights["fc_bias"])
    logit = float(np.dot(fc_w, pooled) + fc_b)
    return _sigmoid(logit)


def _init_ort_session(onnx_path: Path, prefer_cuda: bool = True):
    """Initialize ONNX Runtime session preferring CUDAExecutionProvider then CPUExecutionProvider."""
    if not HAS_ORT or not onnx_path.is_file():
        return None, "numpy"
    try:
        _ort.set_default_logger_severity(3)
        avail = _ort.get_available_providers()
        providers = []
        if prefer_cuda and "CUDAExecutionProvider" in avail:
            providers.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in avail:
            providers.append("CPUExecutionProvider")
        if not providers:
            providers = avail
        sess_opts = _ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        sess = _ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=providers)
        active = sess.get_providers()
        tag = "onnx_cuda" if (active and "CUDA" in active[0]) else "onnx_cpu"
        return sess, tag
    except Exception:
        return None, "numpy"


def predict_rib_hybrid(
    contour_ys: Sequence[float] | np.ndarray | None,
    features: Sequence[float] | np.ndarray | None,
    model: Mapping[str, Any],
    res_scale: float = 1.0,
) -> dict[str, Any] | None:
    """Evaluate rib contour using 1D-CNN (ONNX CUDA/CPU or NumPy) + Mahalanobis Safety Gate."""
    if model is None:
        return None
    mtype = model.get("model_type", "logistic_regression_standard_scaled")
    if mtype == "logistic_regression_standard_scaled":
        if features is None:
            return None
        prob = predict_tip_ng_prob(features, model)
        th = float(model.get("prob_threshold", 0.5))
        return {
            "cnn_prob": prob,
            "mahal_dist": 0.0,
            "mahal_anomaly": False,
            "is_ng": bool(prob >= th),
            "provider": "logreg_numpy",
        }

    sig = extract_contour_1d_signal(contour_ys, res_scale=res_scale)
    if sig is None:
        return None

    norm_mu = np.asarray(model["signal_norm_mean"], dtype=np.float32).reshape(2, 1)
    norm_sd = np.asarray(model["signal_norm_std"], dtype=np.float32).reshape(2, 1)
    safe_sd = np.where(np.abs(norm_sd) < 1e-6, 1.0, norm_sd)
    sig_norm = ((sig - norm_mu) / safe_sd).astype(np.float32)

    sess = model.get("_ort_session")
    provider = model.get("_ort_provider", "numpy")
    if sess is not None:
        try:
            in_name = model.get("_ort_input_name") or sess.get_inputs()[0].name
            logits = sess.run(None, {in_name: sig_norm[None, :, :]})[0]
            cnn_prob = _sigmoid(float(np.asarray(logits).ravel()[0]))
        except Exception:
            cnn_prob = _predict_1dcnn_numpy(sig_norm, model["numpy_weights"])
            provider = "numpy_fallback"
    elif "numpy_weights" in model:
        cnn_prob = _predict_1dcnn_numpy(sig_norm, model["numpy_weights"])
        provider = "numpy"
    else:
        return None

    mahal_dist = 0.0
    mahal_anomaly = False
    mahal_cfg = model.get("mahalanobis")
    if mahal_cfg is not None and features is not None:
        mahal_dist = compute_mahalanobis_distance(features, mahal_cfg)
        mahal_anomaly = bool(mahal_dist > max(1e-6, float(mahal_cfg["threshold"])))

    prob_th = float(model.get("prob_threshold", 0.5))
    return {
        "cnn_prob": float(cnn_prob),
        "mahal_dist": float(mahal_dist),
        "mahal_anomaly": mahal_anomaly,
        "is_ng": bool(cnn_prob >= prob_th or mahal_anomaly),
        "provider": provider,
    }


def load_tip_classifier_model(model_path: str | Path, prefer_cuda: bool = True) -> dict[str, Any] | None:
    """Load serialized tip/contour classifier JSON (LogReg or 1D-CNN+Mahalanobis ONNX/NumPy)."""
    path = Path(model_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        mtype = data.get("model_type", "logistic_regression_standard_scaled")
        if mtype == "1dcnn_mahalanobis_hybrid":
            required = ("signal_norm_mean", "signal_norm_std", "prob_threshold", "numpy_weights")
            if not all(k in data for k in required):
                return None
            onnx_rel = data.get("onnx_file")
            if onnx_rel:
                onnx_path = Path(onnx_rel)
                if not onnx_path.is_absolute():
                    onnx_path = path.parent / onnx_rel
                sess, prov = _init_ort_session(onnx_path, prefer_cuda=prefer_cuda)
                data["_ort_session"] = sess
                data["_ort_provider"] = prov
                if sess is not None:
                    data["_ort_input_name"] = sess.get_inputs()[0].name
            else:
                data["_ort_session"] = None
                data["_ort_provider"] = "numpy"
            return data

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
