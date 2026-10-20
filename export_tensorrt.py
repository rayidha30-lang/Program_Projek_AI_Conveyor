#!/usr/bin/env python3
"""
========================================================================================
🚀 TENSORRT FP16 EXPORTER FOR JETSON ORIN NANO & CUDA GPU
========================================================================================
Deskripsi:
  Mengekspor seluruh model YOLO (best_v2.pt & best_rib_v5.pt) ke format TensorRT (.engine)
  dengan akselerasi presisi setengah (FP16) untuk throughput inferensi maksimal pada Jetson.
========================================================================================
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_TO_EXPORT = [
    os.path.join(BASE_DIR, "Jetson_Vision_Engine", "01_Side_Inspection_Bolt_Pin", "models", "best_v2.pt"),
    os.path.join(BASE_DIR, "Jetson_Vision_Engine", "02_Top_Inspection_Slat_Rib", "models", "best_rib_v5.pt"),
    os.path.join(BASE_DIR, "Vision_Engine_Core", "01_Side_Inspection_Bolt_Pin_Lekukan", "models", "best_v2.pt"),
    os.path.join(BASE_DIR, "Vision_Engine_Core", "02_Top_Inspection_Slat_Rib", "models", "best_rib_v5.pt"),
]


def export_models():
    print("=" * 76)
    print("🚀 NVIDIA TENSORRT FP16 MODEL EXPORTER")
    print("=" * 76)

    try:
        import torch
        if not torch.cuda.is_available():
            print("[!] PERINGATAN: CUDA GPU tidak terdeteksi. Ekspor TensorRT membutuhkan GPU NVIDIA!")
            return False
        device_name = torch.cuda.get_device_name(0)
        print(f"[*] Target Hardware GPU : {device_name} (CUDA Siap)")
    except ImportError:
        print("[!] Modul PyTorch belum terpasang!")
        return False

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] Modul Ultralytics belum terpasang! Jalankan: pip install ultralytics")
        return False

    exported = set()
    for pt_path in MODELS_TO_EXPORT:
        real_path = os.path.abspath(pt_path)
        if not os.path.exists(real_path):
            print(f"[-] Model tidak ditemukan (dilewati): {os.path.relpath(real_path, BASE_DIR)}")
            continue

        if real_path in exported:
            continue
        exported.add(real_path)

        engine_path = real_path.replace(".pt", ".engine")
        print(f"\n[*] Mengonversi: {os.path.basename(real_path)} -> {os.path.basename(engine_path)} (FP16)...")
        print(f"    Path: {real_path}")

        try:
            model = YOLO(real_path)
            model.export(
                format="engine",
                half=True,
                device=0,
                workspace=4,
                verbose=False
            )
            if os.path.exists(engine_path):
                sz_mb = os.path.getsize(engine_path) / (1024 * 1024)
                print(f"[✅] Berhasil dibuat: {os.path.basename(engine_path)} ({sz_mb:.1f} MB)")
            else:
                print(f"[?] Selesai diekspor, periksa file di direktori model.")
        except Exception as e:
            print(f"[❌] Gagal ekspor TensorRT untuk {os.path.basename(real_path)}: {e}")
            print("    Tips: Pastikan TensorRT telah terpasang di sistem (`pip install tensorrt`).")

    print("\n" + "=" * 76)
    print("🎉 Proses Ekspor Selesai!")
    print("=" * 76)
    return True


if __name__ == "__main__":
    export_models()
