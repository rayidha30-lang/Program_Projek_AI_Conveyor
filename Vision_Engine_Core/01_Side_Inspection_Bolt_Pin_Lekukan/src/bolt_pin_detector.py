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
from ultralytics import YOLO


class BoltPinDetector:
    """
    Pendeteksi Baut & Pin berbasis YOLO Object Detection.
    Mendukung deteksi diskrit 'bolt_ok', 'bolt_ng', 'pin_ok', 'pin_ng',
    serta verifikasi jarak relatif centroid antara baut dan pin.
    """
    def __init__(self, model_path, conf_threshold=0.25, max_pin_distance=45.0):
        """
        Inisialisasi Model YOLO.
        :param model_path: Path file bobot .pt YOLO (misal: 'models/best_v2.pt')
        :param conf_threshold: Ambang batas confidence deteksi (0.0 - 1.0)
        :param max_pin_distance: Jarak toleransi maksimum pusat baut ke pin (px)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model YOLO tidak ditemukan di: {model_path}")
        
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.max_pin_distance = max_pin_distance
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}

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

        # Jalankan Inferensi YOLO
        results = self.model.predict(
            source=infer_img,
            conf=self.conf_threshold,
            verbose=False
        )

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
