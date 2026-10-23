"""
==============================================================================
MODUL DETEKSI BAUT & PIN KONVEYOR (BOLT & PIN DETECTOR)
==============================================================================
Deskripsi:
  Modul pendeteksi keberadaan, kelengkapan, dan posisi baut (bolt) serta pin
  pada rantai konveyor samping (CAM-01 / CAM-02) menggunakan model YOLO.
==============================================================================
"""

import os
import cv2
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ultralytics import YOLO


class BoltPinDetector:
    """
    Pendeteksi Baut & Pin berbasis YOLO Object Detection.
    Mendukung deteksi diskrit 'bolt_ok', 'bolt_ng', 'pin_ok', 'pin_ng',
    serta verifikasi jarak relatif centroid antara baut dan pin.
    """
    def __init__(self, model_path, conf_threshold=0.35, iou_threshold=0.45, max_pin_distance=45.0, agnostic_nms=True, device=None):
        """
        Inisialisasi Model YOLO.
        :param model_path: Path file bobot .pt YOLO (misal: 'models/best_v2.pt')
        :param conf_threshold: Ambang batas confidence deteksi (0.0 - 1.0)
        :param iou_threshold: Ambang batas IoU untuk Non-Maximum Suppression (NMS)
        :param max_pin_distance: Jarak toleransi maksimum pusat baut ke pin (px)
        :param agnostic_nms: Mengeliminasi duplikat antar-kelas yang saling tumpang tindih
        :param device: Komputasi AI ('auto', 'cuda', 0, atau 'cpu')
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model YOLO tidak ditemukan di: {model_path}")

        # Konfigurasi Device Komputasi (CUDA / CPU)
        if device is None or str(device).lower() in ("auto", ""):
            if HAS_TORCH and torch.cuda.is_available():
                self.device = 0
            else:
                self.device = "cpu"
        elif str(device).lower() in ("cuda", "gpu"):
            self.device = 0 if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        elif str(device).isdigit():
            self.device = int(device)
        else:
            self.device = str(device)

        if self.device != "cpu" and HAS_TORCH and torch.cuda.is_available():
            dev_idx = self.device if isinstance(self.device, int) else 0
            dev_name = torch.cuda.get_device_name(dev_idx)
            self.device_info = f"CUDA GPU #{dev_idx} ({dev_name})"
            self.use_half = True
        else:
            self.device = "cpu"
            self.device_info = "CPU"
            self.use_half = False

        # Cek apakah ada bobot TensorRT Engine (.engine) yang tersedia untuk akselerasi GPU Orin
        engine_candidate = os.path.splitext(model_path)[0] + ".engine"
        actual_model_to_load = model_path
        if self.device != "cpu" and os.path.exists(engine_candidate):
            actual_model_to_load = engine_candidate
            print(f"[BoltPinDetector] 🚀 Menggunakan TensorRT Engine FP16: {os.path.basename(engine_candidate)}")

        self.actual_model_to_load = actual_model_to_load
        self.model = YOLO(actual_model_to_load)
        if self.device != "cpu" and HAS_TORCH and torch.cuda.is_available() and not actual_model_to_load.endswith((".engine", ".onnx")):
            try:
                self.model.to(self.device)
                if hasattr(self.model, "model") and self.model.model is not None:
                    self.model.model.half()
                    self.use_half = True
            except Exception:
                pass
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_pin_distance = max_pin_distance
        self.agnostic_nms = agnostic_nms
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
        is_bolt_model = any("bolt" in str(v).lower() or "pin" in str(v).lower() for v in self.class_names.values())
        if not is_bolt_model:
            print(f"[BoltPinDetector] ⚠️ PERINGATAN: Model '{os.path.basename(actual_model_to_load)}' memiliki kelas {self.class_names} dan BUKAN model deteksi Baut/Pin!")
            print(f"[BoltPinDetector] 🔄 Otomatis beralih ke bobot model Baut/Pin resmi...")
            cur_d = os.path.dirname(os.path.abspath(__file__))
            p_d = os.path.dirname(cur_d)
            bolt_cands = [
                os.path.join(p_d, "models", "best_v2.engine"),
                os.path.join(p_d, "models", "best_v2.onnx"),
                os.path.join(p_d, "models", "best_v2.pt"),
                os.path.join(p_d, "models", "best_v3.pt"),
                os.path.join(cur_d, "..", "..", "..", "Dataset Yolo", "best_v2.pt")
            ]
            for bc in bolt_cands:
                if os.path.exists(bc):
                    try:
                        self.model = YOLO(bc)
                        if self.device != "cpu" and HAS_TORCH and torch.cuda.is_available() and not bc.endswith((".engine", ".onnx")):
                            self.model.to(self.device)
                        actual_model_to_load = bc
                        self.actual_model_to_load = bc
                        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
                        print(f"[BoltPinDetector] ✅ Berhasil beralih ke Model Baut/Pin: {os.path.basename(bc)} (Kelas: {self.class_names})")
                        break
                    except Exception as fb_e:
                        print(f"[BoltPinDetector] Gagal memuat fallback: {fb_e}")

        print(f"[BoltPinDetector] Model dimuat: {os.path.basename(actual_model_to_load)} (Device: {self.device_info})")

    def detect(self, frame, roi=None):
        """
        Melakukan inferensi deteksi baut & pin pada satu frame.
        :param frame: Frame citra BGR (np.ndarray)
        :param roi: Optional (x1, y1, x2, y2) untuk membatasi zona deteksi
        :return: Dict hasil deteksi terstruktur
        """
        if frame is None or frame.size == 0:
            return {
                "bolts": [],
                "pins": [],
                "pairs": [],
                "status": "INVALID",
                "message": "Frame kosong"
            }

        # Jika ROI diberikan, potong gambar untuk efisiensi
        offset_x, offset_y = 0, 0
        infer_img = frame
        if roi is not None:
            rx1, ry1, rx2, ry2 = [int(v) for v in roi]
            h, w = frame.shape[:2]
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(w, rx2), min(h, ry2)
            if rx2 > rx1 and ry2 > ry1:
                infer_img = frame[ry1:ry2, rx1:rx2]
                offset_x, offset_y = rx1, ry1

        # Jalankan Inferensi YOLO dengan Agnostic NMS untuk mencegah bounding box ganda
        try:
            call_kwargs = {
                "source": infer_img,
                "conf": self.conf_threshold,
                "iou": self.iou_threshold,
                "device": self.device,
                "agnostic_nms": self.agnostic_nms,
                "verbose": False
            }
            is_trt_or_onnx = hasattr(self, "actual_model_to_load") and (str(self.actual_model_to_load).endswith((".engine", ".onnx")))
            if not is_trt_or_onnx:
                call_kwargs["imgsz"] = 640
                if self.use_half and self.device != "cpu":
                    call_kwargs["half"] = True
            results = self.model(**call_kwargs)
        except (RuntimeError, Exception) as e_cuda:
            err_str = str(e_cuda).lower()
            if ("out of memory" in err_str or "cuda" in err_str or "allocation" in err_str) and self.device != "cpu":
                print(f"\n[WARN BOLT-PIN DETECTOR] CUDA Memory Error terdeteksi ({e_cuda}). Beralih otomatis ke komputasi CPU...")
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                self.device = "cpu"
                self.device_info = "CPU (Fallback)"
                self.use_half = False
                try:
                    if hasattr(self.model, "model") and self.model.model is not None:
                        self.model.model.float()
                        self.model.to("cpu")
                except Exception:
                    pass
                results = self.model(
                    source=infer_img,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device="cpu",
                    agnostic_nms=self.agnostic_nms,
                    imgsz=480,
                    verbose=False
                )
            else:
                raise e_cuda

        detected_bolts = []
        detected_pins = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = self.class_names.get(cls_id, str(cls_id)).lower()

                # Konversi koordinat relatif ke global frame
                gx1 = int(xyxy[0] + offset_x)
                gy1 = int(xyxy[1] + offset_y)
                gx2 = int(xyxy[2] + offset_x)
                gy2 = int(xyxy[3] + offset_y)
                cx = int((gx1 + gx2) / 2)
                cy = int((gy1 + gy2) / 2)

                item_data = {
                    "box": [gx1, gy1, gx2, gy2],
                    "centroid": (cx, cy),
                    "confidence": conf,
                    "class_name": cls_name,
                    "is_ng": "ng" in cls_name
                }

                if "bolt" in cls_name:
                    detected_bolts.append(item_data)
                elif "pin" in cls_name:
                    detected_pins.append(item_data)

        # Pasangkan Baut dengan Pin terdekat untuk validasi integritas
        pairs = self._pair_bolts_and_pins(detected_bolts, detected_pins)

        # Evaluasi status keseluruhan deteksi pada frame ini
        overall_status = "OK"
        if len(detected_bolts) == 0 and len(detected_pins) == 0:
            overall_status = "EMPTY"  # Belum ada slat masuk ke gerbang
        else:
            for pair in pairs:
                if pair["status"] == "NG":
                    overall_status = "NG"
                    break

        return {
            "bolts": detected_bolts,
            "pins": detected_pins,
            "pairs": pairs,
            "status": overall_status
        }

    def _pair_bolts_and_pins(self, bolts, pins):
        """
        Memasangkan setiap baut ke pin terdekat untuk memastikan
        tidak ada baut tanpa pin atau pin bergeser.
        """
        pairs = []
        used_pins = set()

        for b_idx, bolt in enumerate(bolts):
            bcx, bcy = bolt["centroid"]
            best_pin = None
            min_dist = float("inf")
            best_pin_idx = -1

            for p_idx, pin in enumerate(pins):
                if p_idx in used_pins:
                    continue
                pcx, pcy = pin["centroid"]
                dist = np.sqrt((bcx - pcx) ** 2 + (bcy - pcy) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_pin = pin
                    best_pin_idx = p_idx

            # Evaluasi status pasangan
            pair_status = "OK"
            reasons = []

            if bolt["is_ng"]:
                pair_status = "NG"
                reasons.append("BOLT_NG")

            if best_pin is not None and min_dist <= self.max_pin_distance:
                used_pins.add(best_pin_idx)
                if best_pin["is_ng"]:
                    pair_status = "NG"
                    reasons.append("PIN_NG")
            else:
                # Pin tidak ditemukan atau jaraknya terlalu jauh (bergeser/lepas)
                pair_status = "NG"
                reasons.append("PIN_MISSING_OR_DISPLACED")

            pairs.append({
                "bolt": bolt,
                "pin": best_pin,
                "distance": min_dist if best_pin else None,
                "status": pair_status,
                "reasons": reasons
            })

        return pairs


# ==============================================================================
# STANDALONE RUNNER: INSPEKSI BAUT & PIN LIVE WEBCAM / VIDEO (TANPA PATCHCORE)
# ==============================================================================
if __name__ == "__main__":
    import argparse
    import sys
    import time

    parser = argparse.ArgumentParser(description="Live Bolt & Pin Inspector (YOLO Only - Tanpa PatchCore)")
    parser.add_argument("--cam", type=int, default=None, help="Indeks kamera webcam (misal: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path ke file video pengujian")
    parser.add_argument("--model", type=str, default=None, help="Path ke model YOLO best.pt")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold YOLO (default: 0.35)")
    args = parser.parse_args()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(cur_dir)

    # 1. Cari path model YOLO
    model_candidates = [
        args.model,
        os.path.join(parent_dir, "models", "best_v2.pt"),
        os.path.join(parent_dir, "models", "best_v3.pt"),
        os.path.join(parent_dir, "..", "..", "Program AI", "1. Inspeksi Bolt & Pin", "3. Model Hybrid (YOLO + PatchCore)", "best_v2.pt")
    ]
    model_path = next((m for m in model_candidates if m and os.path.exists(m)), None)
    if not model_path:
        print("[!] Model YOLO Baut & Pin tidak ditemukan!")
        sys.exit(1)

    print("=" * 70)
    print("🔩 INSPEKSI BAUT & PIN KONVEYOR (YOLO ONLY - VISION ENGINE CORE)")
    print("=" * 70)
    print(f"[*] Model: {os.path.basename(model_path)}")
    detector = BoltPinDetector(model_path, conf_threshold=args.conf)

    # 2. Inisialisasi Sumber Kamera / Video
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    if args.cam is not None:
        cap = cv2.VideoCapture(args.cam, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        source_name = f"Live Webcam #{args.cam}"
    elif args.video and os.path.exists(args.video):
        cap = cv2.VideoCapture(args.video)
        source_name = os.path.basename(args.video)
    else:
        # Coba buka webcam default #0 jika tidak ada argumen
        cap = cv2.VideoCapture(0, backend)
        source_name = "Live Webcam #0"
        if not cap.isOpened():
            print("[!] Tidak ada webcam terdeteksi. Silakan tentukan --video <path>.")
            sys.exit(1)

    print(f"[*] Sumber: {source_name}")
    print("    [SPACE]: Jeda/Lanjut | [Q]/[ESC]: Keluar")
    print("=" * 70)

    is_paused = False
    prev_time = time.time()

    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                if args.cam is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
        
        vis = frame.copy()
        res = detector.detect(vis)
        pairs = res.get("pairs", [])

        # Visualisasi Baut & Pin
        for pair in pairs:
            b = pair["bolt"]
            p = pair["pin"]
            bx1, by1, bx2, by2 = b["box"]
            b_col = (0, 0, 255) if b["is_ng"] else (0, 255, 0)
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), b_col, 2)
            cv2.putText(vis, f"{b['class_name']} {b['confidence']:.2f}", (bx1, max(18, by1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, b_col, 2)

            if p:
                px1, py1, px2, py2 = p["box"]
                p_col = (0, 0, 255) if p["is_ng"] else (0, 255, 200)
                cv2.rectangle(vis, (px1, py1), (px2, py2), p_col, 2)
                cv2.putText(vis, f"{p['class_name']}", (px1, max(18, py1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, p_col, 1)
                # Garis penghubung baut ke pin
                cv2.line(vis, b["centroid"], p["centroid"], (255, 255, 0), 2)
                dist = pair.get("distance")
                if dist:
                    mx, my = (b["centroid"][0] + p["centroid"][0]) // 2, (b["centroid"][1] + p["centroid"][1]) // 2
                    cv2.putText(vis, f"{dist:.1f}px", (mx, my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Hitung FPS
        now = time.time()
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
        prev_time = now

        # Overlay Banner Info
        total_pairs = len(pairs)
        ng_pairs = sum(1 for pr in pairs if pr["status"] == "NG")
        status_txt = "ALL OK" if (total_pairs > 0 and ng_pairs == 0) else (f"{ng_pairs} NG DETECTED" if ng_pairs > 0 else "NO BOLT")
        banner_col = (0, 160, 0) if (total_pairs > 0 and ng_pairs == 0) else ((0, 0, 200) if ng_pairs > 0 else (60, 60, 60))

        cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (20, 20, 20), -1)
        cv2.putText(vis, f"BOLT & PIN YOLO | FPS: {fps:.1f} | {source_name}", (15, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(vis, (vis.shape[1] - 220, 5), (vis.shape[1] - 10, 35), banner_col, -1)
        cv2.putText(vis, status_txt, (vis.shape[1] - 210, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Bolt & Pin Inspector - Vision Engine Core", vis)
        key = cv2.waitKey(1 if not is_paused else 30) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            break
        elif key in [ord(' '), ord('p'), ord('P')]:
            is_paused = not is_paused

    cap.release()
    cv2.destroyAllWindows()
