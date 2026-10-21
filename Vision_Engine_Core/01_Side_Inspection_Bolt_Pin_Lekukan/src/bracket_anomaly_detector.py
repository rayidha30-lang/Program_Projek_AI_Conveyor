"""
==============================================================================
MODUL AUDIT ANOMALI LEKUKAN PLAT / BRACKET (PATCHCORE DETECTOR)
==============================================================================
Deskripsi:
  Modul deteksi keretakan (crack), patah, aus, atau deformasi fisik pada
  lekukan plat samping konveyor (CAM-01 / CAM-02) menggunakan PatchCore
  (Unsupervised Anomaly Detection dengan Pretrained ResNet-18).
==============================================================================
"""

import os
import pickle
import numpy as np
import cv2

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torchvision.models import resnet18, ResNet18_Weights
    from sklearn.neighbors import NearestNeighbors
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PatchCoreFeatureExtractor:
    """
    Ekstraktor fitur spasial tingkat menengah (Layer 2 & Layer 3) ResNet-18.
    """
    def __init__(self, input_size=(224, 224), device=None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch & Torchvision wajib terinstall.")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.input_h, self.input_w = input_size
        self.device = torch.device(device)

        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Identity()
        self.model.eval().to(self.device)

        self._features = {}
        self.model.layer2.register_forward_hook(lambda m, i, o: self._features.__setitem__("layer2", o.detach()))
        self.model.layer3.register_forward_hook(lambda m, i, o: self._features.__setitem__("layer3", o.detach()))

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)

    def extract(self, bgr_image):
        """
        Ekstraksi patch embeddings dari crop citra BGR langsung pada GPU Tensor.
        Output: torch.Tensor shape (N_patches, Feature_Dim) pada GPU device
        """
        if bgr_image is None or bgr_image.size == 0:
            return None

        # Zero-copy OpenCV Resize & Direct Float Tensor (3x lebih cepat dari PIL)
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        rgb_tensor = torch.from_numpy(rgb_resized).to(self.device).permute(2, 0, 1).float().div_(255.0)
        x = rgb_tensor.sub_(self.mean).div_(self.std).unsqueeze(0)

        with torch.inference_mode():
            self._features.clear()
            self.model(x)

            f2 = self._features["layer2"]  # (1, 128, H2, W2)
            f3 = self._features["layer3"]  # (1, 256, H3, W3)
            h_feat, w_feat = f3.shape[-2:]

            # Interpolasi Layer 2 ke ukuran Layer 3 (sesuai spesifikasi training PatchCore)
            f2_down = nn.functional.interpolate(
                f2, size=(h_feat, w_feat), mode="bilinear", align_corners=False
            )
            f2_flat = f2_down.squeeze(0).permute(1, 2, 0).reshape(-1, f2.shape[1])
            f3_flat = f3.squeeze(0).permute(1, 2, 0).reshape(-1, f3.shape[1])

            # Gabungkan fitur layer 2 + 3 (Total dimensi: 128 + 256 = 384) & L2 Normalization
            emb = torch.cat([f2_flat, f3_flat], dim=1)
            emb = nn.functional.normalize(emb, p=2, dim=1)

            return emb, (h_feat, w_feat)


class BracketAnomalyDetector:
    """
    Detektor anomali lekukan plat menggunakan model PatchCore Memory Bank dengan akselerasi GPU murni.
    """
    def __init__(self, model_path, default_thresh=0.8650, device=None):
        """
        :param model_path: Path ke file memory bank .pkl
        :param default_thresh: Ambang batas skor anomali default
        :param device: "cpu" atau "cuda" (default: auto CUDA jika tersedia)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model PatchCore tidak ditemukan di: {model_path}")

        if device is None:
            device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

        self.device = torch.device(device)
        print(f"[*] Memuat model PatchCore dari: {model_path}")
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self.memory_bank = data.get("memory_bank", None)
        self.calibrated_threshold = data.get("threshold", default_thresh)
        if self.calibrated_threshold is None or np.isnan(self.calibrated_threshold):
            self.calibrated_threshold = default_thresh

        print(f"    - Memory Bank Patches : {len(self.memory_bank)}")
        print(f"    - Calibrated Threshold: {self.calibrated_threshold:.4f}")

        # Simpan Memory Bank di PyTorch GPU Tensor untuk akselerasi jarak 180x lebih cepat dari CPU scikit-learn
        self.memory_bank_torch = torch.from_numpy(self.memory_bank).to(self.device).float()

        # Inisialisasi Feature Extractor
        self.extractor = PatchCoreFeatureExtractor(input_size=(224, 224), device=device)

    def inspect_crop(self, crop_bgr, custom_thresh=None):
        """
        Audit kondisi fisik lekukan plat dari gambar crop menggunakan GPU torch.cdist.
        :param crop_bgr: Crop citra area lekukan (np.ndarray BGR)
        :param custom_thresh: Override threshold (opsional)
        :return: dict berisi status, score, heatmap, dan overlay
        """
        if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 10 or crop_bgr.shape[1] < 10:
            return {
                "status": "INVALID",
                "score": 0.0,
                "threshold": self.calibrated_threshold,
                "heatmap": None,
                "overlay": crop_bgr
            }

        thresh = custom_thresh if custom_thresh is not None else self.calibrated_threshold

        try:
            res = self.extractor.extract(crop_bgr)
            if res is None:
                return {"status": "INVALID", "score": 0.0, "threshold": thresh, "heatmap": None, "overlay": crop_bgr}

            embeddings_torch, (h_feat, w_feat) = res

            # Akselerasi GPU 1-NN via torch.cdist (0.7 ms vs 130 ms CPU scikit-learn)
            with torch.inference_mode():
                dists_torch = torch.cdist(embeddings_torch, self.memory_bank_torch).min(dim=1).values
                n_top = max(1, int(dists_torch.shape[0] * 0.03))
                top_vals, _ = torch.topk(dists_torch, n_top)
                anomaly_score = float(top_vals.mean().item())

            # Status Keputusan
            status = "NG" if anomaly_score >= thresh else "OK"

            # Lazy Heatmap generation: hanya dihitung jika status NG (menghemat 20ms CPU per frame normal)
            overlay = crop_bgr
            heatmap_color = None
            if status == "NG":
                h_orig, w_orig = crop_bgr.shape[:2]
                dists_flat = dists_torch.cpu().numpy()
                dist_grid = dists_flat.reshape(h_feat, w_feat)
                dist_resized = cv2.resize(dist_grid, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                norm_map = np.clip((dist_resized - 0.2) / (thresh * 1.3 - 0.2 + 1e-6), 0.0, 1.0)
                heatmap_color = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(crop_bgr, 0.65, heatmap_color, 0.35, 0)

            return {
                "status": status,
                "score": anomaly_score,
                "threshold": float(thresh),
                "heatmap": heatmap_color,
                "overlay": overlay
            }
        except Exception as e:
            return {
                "status": "INVALID",
                "score": 0.0,
                "threshold": thresh,
                "message": str(e),
                "heatmap": None,
                "overlay": crop_bgr
            }
