"""
GPU-Accelerated Vision Operations (TensorRT FP16 + ONNX Runtime CUDA IO-Binding + PyTorch GPU Preprocessing/NMS).

Eliminates CPU letterbox resizing, CPU float32 normalization, and Host<->Device Memcpy nodes
when running Ultralytics-exported YOLOv8/YOLO11/YOLO26 .engine or .onnx models on CUDA GPUs.
"""

import ast
import json
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    import torchvision
    HAS_TORCH_CUDA = bool(torch.cuda.is_available())
except Exception:
    torch = None
    F = None
    torchvision = None
    HAS_TORCH_CUDA = False

try:
    import onnxruntime as ort
    HAS_ORT_CUDA = "CUDAExecutionProvider" in ort.get_available_providers()
except Exception:
    ort = None
    HAS_ORT_CUDA = False

try:
    import tensorrt as trt
    HAS_TRT = True
except Exception:
    trt = None
    HAS_TRT = False


def is_gpu_onnx_supported():
    """Return True when PyTorch CUDA, TorchVision, and ONNX Runtime CUDAExecutionProvider are available."""
    return bool(HAS_TORCH_CUDA and HAS_ORT_CUDA and torchvision is not None)


def is_gpu_trt_supported():
    """Return True when PyTorch CUDA, TorchVision, and TensorRT Python runtime are available."""
    return bool(HAS_TORCH_CUDA and HAS_TRT and torchvision is not None)


def create_gpu_yolo_runner(weights_path, device_id=0):
    """Instantiate GpuTrtYoloRunner (.engine) or GpuOnnxYoloRunner (.onnx) when supported, else None."""
    w_str = str(weights_path or "")
    if w_str.endswith(".engine") and is_gpu_trt_supported():
        return GpuTrtYoloRunner(w_str, device_id=device_id)
    if w_str.endswith(".onnx") and is_gpu_onnx_supported():
        return GpuOnnxYoloRunner(w_str, device_id=device_id)
    return None


class _GpuYoloBase:
    """Shared GPU letterbox preprocessing and GPU NMS postprocessing for YOLO runners."""

    def __init__(self, device_id=0):
        self.device_id = int(device_id)
        self.torch_device = torch.device(f"cuda:{self.device_id}")
        self.in_h = 640
        self.in_w = 640
        self.is_fp16 = False
        self.torch_dtype = torch.float32
        self.pad_value = 114.0 / 255.0
        self.class_names = {}
        self.num_classes = 0
        self.task = "detect"
        self.out_buf = None
        self.decode_seg_masks = True
        self.last_mask_profiles = []

    @staticmethod
    def _normalize_names(raw_names):
        if isinstance(raw_names, dict):
            return {int(k): str(v) for k, v in raw_names.items()}
        if isinstance(raw_names, (list, tuple)):
            return {i: str(v) for i, v in enumerate(raw_names)}
        return {}

    def _preprocess_gpu(self, img_bgr):
        """
        Upload raw uint8 BGR image once to GPU, then perform BGR->RGB, [0,1] scaling,
        aspect-preserving letterbox resize, and constant 114/255 padding entirely on CUDA.
        """
        src_h, src_w = img_bgr.shape[:2]
        scale = min(self.in_h / float(src_h), self.in_w / float(src_w))
        new_h = int(round(src_h * scale))
        new_w = int(round(src_w * scale))

        if not img_bgr.flags["C_CONTIGUOUS"]:
            img_bgr = np.ascontiguousarray(img_bgr)

        t = torch.from_numpy(img_bgr).to(self.torch_device, non_blocking=True)
        t = t[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0)
        t = t.to(dtype=self.torch_dtype).mul_(1.0 / 255.0)

        if new_h != src_h or new_w != src_w:
            t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)

        pad_top = (self.in_h - new_h) // 2
        pad_bot = self.in_h - new_h - pad_top
        pad_left = (self.in_w - new_w) // 2
        pad_right = self.in_w - new_w - pad_left
        if pad_top or pad_bot or pad_left or pad_right:
            t = F.pad(t, (pad_left, pad_right, pad_top, pad_bot), value=self.pad_value)

        return t.contiguous(), scale, pad_top, pad_left

    def _get_proto_tensor(self):
        """Return 4D prototype mask tensor (1, C_m, H_p, W_p) when available from YOLO-seg outputs."""
        for item in getattr(self, "extra_outputs", ()) or ():
            buf = item[-1] if isinstance(item, (tuple, list)) and len(item) >= 4 else None
            if buf is not None and getattr(buf, "ndim", 0) == 4:
                return buf
        for item in getattr(self, "extra_out_bufs", ()) or ():
            buf = item[-1] if isinstance(item, (tuple, list)) and len(item) >= 2 else None
            if buf is not None and getattr(buf, "ndim", 0) == 4:
                return buf
        return None

    def _decode_mask_profiles_gpu(self, mask_coeffs_kept, lb_xyxy_kept, scale, pad_top, pad_left, src_h, src_w):
        """Decode prototype segmentation masks on GPU for NMS-surviving boxes and extract 1D rib column envelopes."""
        self.last_mask_profiles = []
        if not getattr(self, "decode_seg_masks", True):
            return
        if mask_coeffs_kept is None or mask_coeffs_kept.numel() == 0 or lb_xyxy_kept is None or lb_xyxy_kept.numel() == 0:
            return
        proto_buf = self._get_proto_tensor()
        if proto_buf is None:
            return
        proto = proto_buf[0]
        if proto.ndim != 3:
            return
        c_m, h_p, w_p = int(proto.shape[0]), int(proto.shape[1]), int(proto.shape[2])
        if mask_coeffs_kept.shape[1] != c_m or h_p <= 0 or w_p <= 0:
            return

        masks_proto = torch.sigmoid(mask_coeffs_kept.float() @ proto.float().view(c_m, -1)).view(-1, h_p, w_p)
        lb_boxes_np = lb_xyxy_kept.detach().cpu().numpy()
        p_sx = float(w_p) / max(1.0, float(self.in_w))
        p_sy = float(h_p) / max(1.0, float(self.in_h))
        inv_scale = 1.0 / max(1e-6, float(scale))
        max_w = float(src_w) if src_w is not None else float(self.in_w)
        max_h = float(src_h) if src_h is not None else float(self.in_h)

        profiles = []
        for idx in range(len(lb_boxes_np)):
            lx1, ly1, lx2, ly2 = lb_boxes_np[idx]
            px1 = int(np.clip(np.floor(lx1 * p_sx), 0, w_p - 1))
            px2 = int(np.clip(np.ceil(lx2 * p_sx), px1 + 1, w_p))
            py1 = int(np.clip(np.floor(ly1 * p_sy), 0, h_p - 1))
            py2 = int(np.clip(np.ceil(ly2 * p_sy), py1 + 1, h_p))

            prob_thr = 0.45
            sub_prob_np = masks_proto[idx, py1:py2, px1:px2].detach().cpu().numpy().astype(np.float32, copy=False)
            sub_m_np = sub_prob_np >= prob_thr
            if sub_m_np.size == 0 or not np.any(sub_m_np):
                profiles.append(None)
                continue

            col_active = np.any(sub_m_np, axis=0)
            act_cols = np.where(col_active)[0]
            if len(act_cols) < 4:
                profiles.append(None)
                continue

            seg_x1 = float(np.clip(((px1 + float(act_cols[0])) / p_sx - pad_left) * inv_scale, 0.0, max(0.0, max_w - 1.0)))
            seg_x2 = float(np.clip(((px1 + float(act_cols[-1]) + 1.0) / p_sx - pad_left) * inv_scale, seg_x1 + 1.0, max_w))

            sub_act = sub_m_np[:, act_cols]
            prob_act = sub_prob_np[:, act_cols]
            h_sub = int(sub_act.shape[0])
            col_idx = np.arange(len(act_cols))

            # Zero-crossing sub-pixel interpolation across prob_thr to avoid 8px/12px staircase quantization
            top_rows = np.argmax(sub_act, axis=0)
            top_sub = top_rows.astype(np.float32) + 0.5
            top_has_prev = top_rows > 0
            if np.any(top_has_prev):
                r_hi = top_rows[top_has_prev]
                c_hi = col_idx[top_has_prev]
                p_hi = prob_act[r_hi, c_hi]
                p_lo = prob_act[r_hi - 1, c_hi]
                valid_zc = p_lo < prob_thr
                if np.any(valid_zc):
                    denom_top = np.maximum(p_hi[valid_zc] - p_lo[valid_zc], 1e-4)
                    frac_top = np.clip((prob_thr - p_lo[valid_zc]) / denom_top, 0.0, 1.0)
                    top_vals = top_sub[top_has_prev]
                    top_vals[valid_zc] = (r_hi[valid_zc].astype(np.float32) - 0.5) + frac_top
                    top_sub[top_has_prev] = top_vals

            bot_rows = h_sub - 1 - np.argmax(sub_act[::-1, :], axis=0)
            bot_sub = bot_rows.astype(np.float32) + 0.5
            bot_has_next = bot_rows < (h_sub - 1)
            if np.any(bot_has_next):
                r_lo = bot_rows[bot_has_next]
                c_lo = col_idx[bot_has_next]
                p_in = prob_act[r_lo, c_lo]
                p_out = prob_act[r_lo + 1, c_lo]
                valid_zc_b = p_out < prob_thr
                if np.any(valid_zc_b):
                    denom_bot = np.maximum(p_in[valid_zc_b] - p_out[valid_zc_b], 1e-4)
                    frac_bot = np.clip((p_in[valid_zc_b] - prob_thr) / denom_bot, 0.0, 1.0)
                    bot_vals = bot_sub[bot_has_next]
                    bot_vals[valid_zc_b] = (r_lo[valid_zc_b].astype(np.float32) + 0.5) + frac_bot
                    bot_sub[bot_has_next] = bot_vals

            if len(act_cols) >= 5:
                top_pad = np.pad(top_sub, 1, mode="edge")
                bot_pad = np.pad(bot_sub, 1, mode="edge")
                top_sub = np.median(np.stack([top_pad[:-2], top_pad[1:-1], top_pad[2:]], axis=0), axis=0).astype(np.float32)
                bot_sub = np.median(np.stack([bot_pad[:-2], bot_pad[1:-1], bot_pad[2:]], axis=0), axis=0).astype(np.float32)

            col_xs = ((px1 + act_cols.astype(np.float32) + 0.5) / p_sx - pad_left) * inv_scale
            top_ys = ((py1 + top_sub) / p_sy - pad_top) * inv_scale
            bot_ys = ((py1 + bot_sub) / p_sy - pad_top) * inv_scale

            col_xs = np.clip(col_xs, 0.0, max_w).astype(np.float32)
            top_ys = np.clip(top_ys, 0.0, max_h).astype(np.float32)
            bot_ys = np.clip(bot_ys, 0.0, max_h).astype(np.float32)

            profiles.append({
                "seg_x1": seg_x1,
                "seg_x2": seg_x2,
                "col_xs": col_xs,
                "top_ys": top_ys,
                "bot_ys": bot_ys,
            })

        self.last_mask_profiles = profiles

    def _postprocess_gpu(self, pred_tensor, conf_thresh, iou_thresh, agnostic_nms, scale, pad_top, pad_left, src_h=None, src_w=None):
        """Filter confidence scores and apply NMS on GPU, copying only surviving boxes (and seg mask envelopes) to CPU."""
        self.last_mask_profiles = []
        pred_raw = pred_tensor[0]
        proto_buf = self._get_proto_tensor()
        mask_dim = int(proto_buf.shape[1]) if (proto_buf is not None and proto_buf.ndim == 4) else 0
        is_end2end = bool(
            pred_raw.ndim == 2
            and pred_raw.shape[0] > pred_raw.shape[1]
            and ((mask_dim > 0 and pred_raw.shape[1] == 6 + mask_dim) or (mask_dim == 0 and pred_raw.shape[1] == 6))
        )

        pred = pred_raw.T if (pred_raw.ndim == 2 and pred_raw.shape[0] < pred_raw.shape[1]) else pred_raw

        if pred.shape[1] <= 4:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

        if is_end2end:
            confs = pred[:, 4].float()
            mask = confs >= float(conf_thresh)
            if not torch.any(mask):
                return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
            xyxy = pred[mask, :4].float()
            c = confs[mask]
            cls_f = pred[mask, 5].long()
            mask_coeffs = pred[mask, 6:] if pred.shape[1] > 6 else None
        else:
            # Slice strictly [4 : 4 + num_classes] so YOLO-seg mask coefficients (e.g. 32 dims) are never treated as classes.
            nc = int(getattr(self, "num_classes", 0) or 0)
            if nc <= 0 and mask_dim > 0 and pred.shape[1] > (4 + mask_dim):
                nc = pred.shape[1] - 4 - mask_dim
            if nc <= 0 and str(getattr(self, "task", "detect")).lower() == "segment" and pred.shape[1] > 36:
                nc = pred.shape[1] - 36
            score_end = min(pred.shape[1], 4 + nc) if nc > 0 else pred.shape[1]
            scores = pred[:, 4:score_end]
            confs, cls_ids = scores.max(dim=1)
            mask = confs >= float(conf_thresh)
            if not torch.any(mask):
                return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

            b = pred[mask, :4].float()
            c = confs[mask].float()
            cls_f = cls_ids[mask]
            mask_coeffs = pred[mask, score_end:] if pred.shape[1] > score_end else None

            half_w = b[:, 2] * 0.5
            half_h = b[:, 3] * 0.5
            xyxy = torch.stack((b[:, 0] - half_w, b[:, 1] - half_h, b[:, 0] + half_w, b[:, 1] + half_h), dim=1)

        if agnostic_nms:
            keep = torchvision.ops.nms(xyxy, c, float(iou_thresh))
        else:
            max_wh = 7680.0
            offsets = cls_f.float().unsqueeze(1) * max_wh
            keep = torchvision.ops.nms(xyxy + offsets, c, float(iou_thresh))

        if keep.numel() == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

        lb_xyxy_kept = xyxy[keep]
        if mask_coeffs is not None and proto_buf is not None:
            self._decode_mask_profiles_gpu(
                mask_coeffs[keep], lb_xyxy_kept, scale, pad_top, pad_left, src_h, src_w
            )

        xyxy_kept = lb_xyxy_kept.clone()
        inv_scale = 1.0 / scale
        xyxy_kept[:, 0] = (xyxy_kept[:, 0] - pad_left) * inv_scale
        xyxy_kept[:, 1] = (xyxy_kept[:, 1] - pad_top) * inv_scale
        xyxy_kept[:, 2] = (xyxy_kept[:, 2] - pad_left) * inv_scale
        xyxy_kept[:, 3] = (xyxy_kept[:, 3] - pad_top) * inv_scale

        return (
            xyxy_kept.cpu().numpy(),
            c[keep].cpu().numpy(),
            cls_f[keep].cpu().numpy(),
        )


class GpuOnnxYoloRunner(_GpuYoloBase):
    """Zero-CPU-preprocessing ONNX Runtime CUDA runner with GPU letterbox, IO-binding, and GPU NMS."""

    def __init__(self, onnx_path, device_id=0):
        if not is_gpu_onnx_supported():
            raise RuntimeError("GPU ONNX Runtime (CUDAExecutionProvider + PyTorch CUDA + torchvision) tidak tersedia.")
        super().__init__(device_id=device_id)

        self.onnx_path = onnx_path
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = [
            ("CUDAExecutionProvider", {"device_id": self.device_id}),
            "CPUExecutionProvider",
        ]
        self.session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
        active_providers = self.session.get_providers()
        if "CUDAExecutionProvider" not in active_providers:
            raise RuntimeError(f"ONNX Runtime gagal mengaktifkan CUDAExecutionProvider (aktif: {active_providers})")

        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        self.input_name = inp.name
        self.output_name = out.name

        inp_shape = inp.shape
        self.in_h = int(inp_shape[2]) if len(inp_shape) >= 4 and isinstance(inp_shape[2], int) else 640
        self.in_w = int(inp_shape[3]) if len(inp_shape) >= 4 and isinstance(inp_shape[3], int) else 640

        self.is_fp16 = "float16" in str(inp.type).lower()
        self.torch_dtype = torch.float16 if self.is_fp16 else torch.float32
        self.np_dtype = np.float16 if self.is_fp16 else np.float32

        out_shape = [int(d) if isinstance(d, int) and d > 0 else 1 for d in out.shape]
        self.out_shape = tuple(out_shape)
        self.out_is_fp16 = "float16" in str(out.type).lower()
        self.out_torch_dtype = torch.float16 if self.out_is_fp16 else torch.float32
        self.out_np_dtype = np.float16 if self.out_is_fp16 else np.float32
        self.out_buf = torch.empty(self.out_shape, dtype=self.out_torch_dtype, device=self.torch_device)
        self.extra_outputs = []
        for extra_out in self.session.get_outputs()[1:]:
            ex_shape = tuple(int(d) if isinstance(d, int) and d > 0 else 1 for d in extra_out.shape)
            ex_dtype = torch.float16 if "float16" in str(extra_out.type).lower() else torch.float32
            ex_np = np.float16 if ex_dtype == torch.float16 else np.float32
            ex_buf = torch.empty(ex_shape, dtype=ex_dtype, device=self.torch_device)
            self.extra_outputs.append((extra_out.name, ex_np, ex_shape, ex_buf))

        self.class_names = self._parse_names_metadata()
        self.num_classes = len(self.class_names)
        self.task = self._parse_task_metadata()

    def _parse_names_metadata(self):
        try:
            meta = self.session.get_modelmeta().custom_metadata_map or {}
            raw = meta.get("names")
            if raw:
                return self._normalize_names(ast.literal_eval(raw))
        except Exception:
            pass
        return {}

    def _parse_task_metadata(self):
        try:
            meta = self.session.get_modelmeta().custom_metadata_map or {}
            return str(meta.get("task", "detect") or "detect").strip().lower()
        except Exception:
            return "detect"

    def predict(self, img_bgr, conf_thresh=0.25, iou_thresh=0.45, agnostic_nms=True):
        """Run GPU-native ONNX inference + GPU NMS."""
        self.last_mask_profiles = []
        if img_bgr is None or img_bgr.size == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

        src_h, src_w = img_bgr.shape[:2]
        with torch.inference_mode():
            inp_tensor, scale, pad_top, pad_left = self._preprocess_gpu(img_bgr)
            torch.cuda.current_stream(self.device_id).synchronize()

            io = self.session.io_binding()
            io.bind_input(
                self.input_name,
                "cuda",
                self.device_id,
                self.np_dtype,
                tuple(inp_tensor.shape),
                inp_tensor.data_ptr(),
            )
            io.bind_output(
                self.output_name,
                "cuda",
                self.device_id,
                self.out_np_dtype,
                self.out_shape,
                self.out_buf.data_ptr(),
            )
            for ex_name, ex_np, ex_shape, ex_buf in self.extra_outputs:
                io.bind_output(
                    ex_name,
                    "cuda",
                    self.device_id,
                    ex_np,
                    ex_shape,
                    ex_buf.data_ptr(),
                )
            self.session.run_with_iobinding(io)
            return self._postprocess_gpu(
                self.out_buf, conf_thresh, iou_thresh, agnostic_nms, scale, pad_top, pad_left, src_h=src_h, src_w=src_w
            )


class GpuTrtYoloRunner(_GpuYoloBase):
    """Zero-CPU-preprocessing TensorRT FP16 runner with GPU letterbox, async CUDA stream, and GPU NMS."""

    def __init__(self, engine_path, device_id=0):
        if not is_gpu_trt_supported():
            raise RuntimeError("TensorRT Python runtime + PyTorch CUDA + torchvision tidak tersedia.")
        super().__init__(device_id=device_id)

        self.engine_path = engine_path
        self._trt_logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(self._trt_logger)

        with open(engine_path, "rb") as f:
            raw = f.read()

        # Ultralytics prefixes TensorRT engines with a 4-byte LE length + JSON metadata header
        plan = raw
        if len(raw) > 8:
            try:
                meta_len = int.from_bytes(raw[:4], byteorder="little")
                if 2 <= meta_len <= 1048576 and (4 + meta_len) < len(raw):
                    meta = json.loads(raw[4:4 + meta_len].decode("utf-8"))
                    if isinstance(meta, dict):
                        self.class_names = self._normalize_names(meta.get("names"))
                        self.num_classes = len(self.class_names)
                        self.task = str(meta.get("task", "detect") or "detect").strip().lower()
                        plan = raw[4 + meta_len:]
            except Exception:
                plan = raw

        self.engine = self._runtime.deserialize_cuda_engine(plan)
        if self.engine is None:
            raise RuntimeError(f"Gagal mendeserialisasi TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self._stream = torch.cuda.Stream(device=self.torch_device)

        self.input_name = None
        self.output_name = None
        self.extra_output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT and self.input_name is None:
                self.input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                if self.output_name is None:
                    self.output_name = name
                else:
                    self.extra_output_names.append(name)

        if not self.input_name or not self.output_name:
            raise RuntimeError(f"Tensor I/O TensorRT tidak lengkap pada {engine_path}")

        inp_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.in_h = int(inp_shape[2]) if len(inp_shape) >= 4 and inp_shape[2] > 0 else 640
        self.in_w = int(inp_shape[3]) if len(inp_shape) >= 4 and inp_shape[3] > 0 else 640

        in_dtype = self.engine.get_tensor_dtype(self.input_name)
        out_dtype = self.engine.get_tensor_dtype(self.output_name)
        self.is_fp16 = (in_dtype == trt.DataType.HALF)
        self.torch_dtype = torch.float16 if self.is_fp16 else torch.float32

        out_torch_dtype = torch.float16 if (out_dtype == trt.DataType.HALF) else torch.float32
        out_shape = tuple(int(d) if d > 0 else 1 for d in self.engine.get_tensor_shape(self.output_name))
        self.out_shape = out_shape
        self.out_buf = torch.empty(self.out_shape, dtype=out_torch_dtype, device=self.torch_device)
        self.extra_out_bufs = []
        for ex_name in self.extra_output_names:
            ex_dtype = self.engine.get_tensor_dtype(ex_name)
            ex_torch_dtype = torch.float16 if (ex_dtype == trt.DataType.HALF) else torch.float32
            ex_shape = tuple(int(d) if d > 0 else 1 for d in self.engine.get_tensor_shape(ex_name))
            self.extra_out_bufs.append((ex_name, torch.empty(ex_shape, dtype=ex_torch_dtype, device=self.torch_device)))

    def predict(self, img_bgr, conf_thresh=0.25, iou_thresh=0.45, agnostic_nms=True):
        """Run GPU-native TensorRT inference + GPU NMS on dedicated non-default CUDA stream."""
        self.last_mask_profiles = []
        if img_bgr is None or img_bgr.size == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

        src_h, src_w = img_bgr.shape[:2]
        with torch.inference_mode():
            cur_stream = torch.cuda.current_stream(self.device_id)
            run_stream = cur_stream if cur_stream.cuda_stream != 0 else self._stream
            with torch.cuda.stream(run_stream):
                inp_tensor, scale, pad_top, pad_left = self._preprocess_gpu(img_bgr)
                self.context.set_tensor_address(self.input_name, int(inp_tensor.data_ptr()))
                self.context.set_tensor_address(self.output_name, int(self.out_buf.data_ptr()))
                for ex_name, ex_buf in self.extra_out_bufs:
                    self.context.set_tensor_address(ex_name, int(ex_buf.data_ptr()))
                self.context.execute_async_v3(int(run_stream.cuda_stream))
                return self._postprocess_gpu(
                    self.out_buf, conf_thresh, iou_thresh, agnostic_nms, scale, pad_top, pad_left, src_h=src_h, src_w=src_w
                )
