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
    Ekstraktor fitur spasial tingkat menengah (Layer 2 & Layer 3) ResNet-18
    dengan akselerasi penuh GPU FP16 dan preprocessing cepat OpenCV.
    """
    def __init__(self, input_size=(224, 224), device=None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch & Torchvision wajib terinstall.")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.input_h, self.input_w = input_size
        self.device = torch.device(device)
        self.half_precision = (self.device.type == "cuda")

        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Identity()
        self.model.eval().to(self.device)
        if self.half_precision:
            self.model.half()

        self._features = {}
        self.model.layer2.register_forward_hook(lambda m, i, o: self._features.__setitem__("layer2", o.detach()))
        self.model.layer3.register_forward_hook(lambda m, i, o: self._features.__setitem__("layer3", o.detach()))

        # Pre-alokasi tensor mean & std ImageNet langsung di GPU untuk eliminasi overhead CPU transform
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.mean_tensor = mean.half() if self.half_precision else mean.float()
        self.std_tensor = std.half() if self.half_precision else std.float()

    def extract(self, bgr_image):
        """
        Ekstraksi patch embeddings dari crop citra BGR langsung di GPU.
        Output: (emb_tensor, (h_feat, w_feat))
        """
        if bgr_image is None or bgr_image.size == 0:
            return None

        # Vektor C++ OpenCV resize + BGR2RGB jauh lebih kencang dibanding PIL Image
        resized_bgr = cv2.resize(bgr_image, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

        with torch.inference_mode():
            t = torch.from_numpy(rgb).to(self.device)
            t = t.half() if self.half_precision else t.float()
            # [H, W, C] -> [1, C, H, W], skala 0..1 dan normalisasi langsung di GPU
            x = (t.permute(2, 0, 1).unsqueeze(0) / 255.0 - self.mean_tensor) / self.std_tensor

            self._features.clear()
            self.model(x)

            f2 = self._features["layer2"]  # (1, 128, H2, W2)
            f3 = self._features["layer3"]  # (1, 256, H3, W3)

            # Interpolasi Layer 2 ke ukuran Layer 3 (sesuai spesifikasi training PatchCore)
            f2_down = nn.functional.interpolate(
                f2, size=f3.shape[-2:], mode="bilinear", align_corners=False
            )
            f2_flat = f2_down.squeeze(0).permute(1, 2, 0).reshape(-1, f2.shape[1])
            f3_flat = f3.squeeze(0).permute(1, 2, 0).reshape(-1, f3.shape[1])

            # Gabungkan fitur layer 2 + 3 (Total dimensi: 128 + 256 = 384) & L2 Normalization
            emb = torch.cat([f2_flat, f3_flat], dim=1)
            emb = nn.functional.normalize(emb, p=2, dim=1)

            h_feat, w_feat = f3.shape[-2:]
            return emb, (h_feat, w_feat)


class BracketAnomalyDetector:
    """
    Detektor anomali lekukan plat menggunakan model PatchCore Memory Bank dengan akselerasi GPU.
    """
    def __init__(self, model_path, default_thresh=0.8650, device=None):
        """
        :param model_path: Path ke file memory bank .pkl
        :param default_thresh: Ambang batas skor anomali default
        :param device: "cpu" atau "cuda" (default: auto CUDA jika tersedia)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model PatchCore tidak ditemukan di: {model_path}")

        if device == "cuda" and (not TORCH_AVAILABLE or not torch.cuda.is_available()):
            print("[WARN BracketAnomalyDetector] CUDA tidak tersedia pada build PyTorch saat ini. Beralih ke CPU.")
            device = "cpu"
        elif device is None:
            device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

        print(f"[*] Memuat model PatchCore dari: {model_path}")
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self.memory_bank = data.get("memory_bank", None)
        self.calibrated_threshold = data.get("threshold", default_thresh)
        if self.calibrated_threshold is None or np.isnan(self.calibrated_threshold):
            self.calibrated_threshold = default_thresh

        print(f"    - Memory Bank Patches : {len(self.memory_bank)}")
        print(f"    - Calibrated Threshold: {self.calibrated_threshold:.4f}")

        # Inisialisasi Feature Extractor
        self.extractor = PatchCoreFeatureExtractor(input_size=(224, 224), device=device)
        self.device = self.extractor.device
        self.use_gpu = (self.device.type == "cuda")

        if self.use_gpu:
            # Simpan memory bank langsung di GPU VRAM untuk pencarian k-NN sub-milidetik (torch.cdist FP16)
            mb_tensor = torch.from_numpy(self.memory_bank).to(self.device)
            self.memory_bank_gpu = mb_tensor.half() if self.extractor.half_precision else mb_tensor.float()
            self.nn_searcher = None
        else:
            self.memory_bank_gpu = None
            self.nn_searcher = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="minkowski", p=2)
            self.nn_searcher.fit(self.memory_bank)

    def inspect_crop(self, crop_bgr, custom_thresh=None):
        """
        Audit kondisi fisik lekukan plat dari gambar crop.
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

            embeddings, (h_feat, w_feat) = res

            if self.use_gpu and self.memory_bank_gpu is not None:
                with torch.inference_mode():
                    # Akselerasi GPU CUDA: Hitung jarak terdekat setiap patch ke Memory Bank (sub-milidetik)
                    dists_matrix = torch.cdist(embeddings, self.memory_bank_gpu, p=2.0)
                    min_dists, _ = torch.min(dists_matrix, dim=1)  # (N_patches,)

                    # Skor Anomali Global = Rata-rata dari top 3% patch jarak terbesar
                    n_top = max(1, int(len(min_dists) * 0.03))
                    top_dists, _ = torch.topk(min_dists, k=n_top)
                    anomaly_score = float(torch.mean(top_dists.float()).item())

                    # Pindahkan ke CPU numpy hanya untuk rendering visual heatmap
                    dists_flat = min_dists.cpu().float().numpy()
            else:
                emb_np = embeddings.cpu().float().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
                distances, _ = self.nn_searcher.kneighbors(emb_np)
                dists_flat = distances[:, 0].astype(np.float32)

                n_top = max(1, int(len(dists_flat) * 0.03))
                top_dists = np.sort(dists_flat)[-n_top:]
                anomaly_score = float(np.mean(top_dists))

            # Bangun Heatmap Anomali
            h_orig, w_orig = crop_bgr.shape[:2]
            dist_grid = dists_flat.reshape(h_feat, w_feat)
            dist_resized = cv2.resize(dist_grid, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
            
            # Normalisasi visualisasi heatmap
            norm_map = np.clip((dist_resized - 0.2) / (thresh * 1.3 - 0.2 + 1e-6), 0.0, 1.0)
            heatmap_color = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

            # Blend dengan citra asli
            overlay = cv2.addWeighted(crop_bgr, 0.65, heatmap_color, 0.35, 0)

            # Status Keputusan
            status = "NG" if anomaly_score > thresh else "OK"

            return {
                "status": status,
                "score": anomaly_score,
                "threshold": thresh,
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

