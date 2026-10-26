#!/usr/bin/env python3
"""
========================================================================================
🚀 TENSORRT FP16 MODEL EXPORTER & BENCHMARK UTILITY FOR JETSON ORIN NANO
========================================================================================
Deskripsi:
  Mengekspor model YOLO resmi (.pt) ke format NVIDIA TensorRT FP16 (.engine)
  secara otomatis untuk memaksimalkan throughput inferensi (target: 45-60+ FPS)
  pada Jetson Orin Nano / CUDA GPU.

Penggunaan:
  1. Ekspor semua model resmi secara otomatis:
     python scripts/export_tensorrt.py

  2. Ekspor model tertentu dengan benchmark performa:
     python scripts/export_tensorrt.py --model "Vision_Engine_Core/02_Top_Inspection_Slat_Rib/models/best_rib_v5.pt" --benchmark
========================================================================================
"""

import os
import sys
import time
import argparse

# Resolusi path absolut direktori root proyek
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_MODELS = [
    os.path.join(PROJECT_ROOT, "Vision_Engine_Core", "02_Top_Inspection_Slat_Rib", "models", "best_rib_v5.pt"),
    os.path.join(PROJECT_ROOT, "Vision_Engine_Core", "01_Side_Inspection_Bolt_Pin_Lekukan", "models", "best_v26_ok.pt"),
    os.path.join(PROJECT_ROOT, "Vision_Engine_Core", "01_Side_Inspection_Bolt_Pin_Lekukan", "models", "best_v26n.pt"),
    os.path.join(PROJECT_ROOT, "Vision_Engine_Core", "01_Side_Inspection_Bolt_Pin_Lekukan", "models", "best_v2.pt"),
]


def check_cuda_and_tensorrt():
    """Memeriksa kesiapan driver CUDA dan modul TensorRT di environment sistem."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("[!] PERINGATAN: CUDA GPU tidak terdeteksi. Ekspor TensorRT membutuhkan GPU NVIDIA!")
            return False, None
        device_name = torch.cuda.get_device_name(0)
        return True, device_name
    except ImportError:
        print("[!] Modul PyTorch belum terpasang!")
        return False, None


def benchmark_engine(engine_path, pt_path=None, imgsz=640, device=0, num_runs=30):
    """Menguji perbandingan latensi & FPS antara PyTorch .pt dan TensorRT .engine."""
    from ultralytics import YOLO
    import numpy as np

    print(f"\n⚡ Menjalankan Verifikasi Benchmark Inferensi ({num_runs} frames, imgsz={imgsz})...")
    dummy_frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

    # 1. Benchmark TensorRT Engine
    try:
        trt_model = YOLO(engine_path, task="detect")
        # Warmup GPU
        for _ in range(5):
            _ = trt_model(dummy_frame, verbose=False)

        t_start = time.perf_counter()
        for _ in range(num_runs):
            _ = trt_model(dummy_frame, verbose=False)
        t_total = time.perf_counter() - t_start
        trt_fps = num_runs / t_total
        trt_lat = (t_total / num_runs) * 1000.0

        print(f"  • [TensorRT Engine] Latensi : {trt_lat:5.1f} ms | Throughput: {trt_fps:5.1f} FPS 🚀")
    except Exception as e:
        print(f"  [!] Gagal menjalankan benchmark TensorRT: {e}")
        return

    # 2. Benchmark PyTorch .pt (sebagai pembanding)
    if pt_path and os.path.exists(pt_path):
        try:
            pt_model = YOLO(pt_path)
            for _ in range(5):
                _ = pt_model(dummy_frame, device=device, verbose=False)

            t_start = time.perf_counter()
            for _ in range(num_runs):
                _ = pt_model(dummy_frame, device=device, verbose=False)
            t_total = time.perf_counter() - t_start
            pt_fps = num_runs / t_total
            pt_lat = (t_total / num_runs) * 1000.0

            speedup = trt_fps / pt_fps if pt_fps > 0 else 1.0
            print(f"  • [PyTorch Base   ] Latensi : {pt_lat:5.1f} ms | Throughput: {pt_fps:5.1f} FPS")
            print(f"  👉 TensorRT FP16 Acceleration : {speedup:.2f}x LEBIH CEPAT!")
        except Exception:
            pass


def export_single_model(pt_path, imgsz=640, half=True, device=0, workspace=4, run_bench=False):
    """Mengekspor satu model YOLO ke format TensorRT FP16 engine."""
    from ultralytics import YOLO

    real_pt = os.path.abspath(pt_path)
    if not os.path.exists(real_pt):
        print(f"[-] Model tidak ditemukan: {os.path.relpath(real_pt, PROJECT_ROOT)}")
        return False

    engine_path = os.path.splitext(real_pt)[0] + ".engine"
    print("\n" + "-" * 76)
    print(f"📦 Mengonversi: {os.path.basename(real_pt)} -> {os.path.basename(engine_path)}")
    print(f"   Lokasi File : {os.path.relpath(real_pt, PROJECT_ROOT)}")
    print(f"   Parameter   : FP16={half}, ImgSz={imgsz}, Device=CUDA:{device}, Workspace={workspace}GB")
    print("-" * 76)

    try:
        model = YOLO(real_pt)
        export_res = model.export(
            format="engine",
            half=half,
            imgsz=imgsz,
            device=device,
            workspace=workspace,
            verbose=False
        )

        # Cari file engine yang berhasil dibuat
        target_engine = None
        if isinstance(export_res, str) and os.path.exists(export_res):
            target_engine = export_res
        elif os.path.exists(engine_path):
            target_engine = engine_path

        if target_engine and os.path.exists(target_engine):
            sz_mb = os.path.getsize(target_engine) / (1024 * 1024)
            print(f"[✅] Berhasil membuat TensorRT Engine: {os.path.basename(target_engine)} ({sz_mb:.1f} MB)")
            if run_bench:
                benchmark_engine(target_engine, pt_path=real_pt, imgsz=imgsz, device=device)
            return True
        else:
            print(f"[!] File engine tidak ditemukan setelah ekspor selesai.")
            return False

    except Exception as e:
        print(f"[❌] Gagal mengekspor model ke TensorRT: {e}")
        print("    Tips untuk Jetson:")
        print("    1. Pastikan paket TensorRT terpasang: python3 -c 'import tensorrt; print(tensorrt.__version__)'")
        print("    2. Jika belum, instal via apt: sudo apt-get install python3-libnvinfer-dev")
        return False


def main():
    parser = argparse.ArgumentParser(description="TensorRT FP16 Model Exporter for Jetson Orin Nano")
    parser.add_argument("--model", type=str, default=None, help="Path ke file model .pt tertentu")
    parser.add_argument("--imgsz", type=int, default=640, help="Ukuran input citra (default: 640)")
    parser.add_argument("--no-half", action="store_true", help="Nonaktifkan presisi FP16 (gunakan FP32)")
    parser.add_argument("--device", type=int, default=0, help="CUDA Device ID (default: 0)")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT Workspace limit dalam GiB (default: 4)")
    parser.add_argument("--benchmark", action="store_true", default=True, help="Jalankan uji latensi/FPS setelah ekspor selesai")
    args = parser.parse_args()

    print("=" * 76)
    print("🚀 NVIDIA TENSORRT FP16 MODEL EXPORTER (JETSON ORIN / CUDA)")
    print("=" * 76)

    cuda_ok, dev_name = check_cuda_and_tensorrt()
    if not cuda_ok:
        sys.exit(1)
    print(f"[*] Target GPU       : {dev_name} (CUDA Siap)")
    print(f"[*] Workspace Limit  : {args.workspace} GiB")
    print(f"[*] Precision Mode   : {'FP16 (Half)' if not args.no_half else 'FP32'}")

    try:
        from ultralytics import YOLO  # noqa
    except ImportError:
        print("[!] Ultralytics belum terpasang. Jalankan: pip install ultralytics")
        sys.exit(1)

    targets = [args.model] if args.model else DEFAULT_MODELS
    success_count = 0
    total_count = 0

    for m in targets:
        if m:
            total_count += 1
            ok = export_single_model(
                pt_path=m,
                imgsz=args.imgsz,
                half=not args.no_half,
                device=args.device,
                workspace=args.workspace,
                run_bench=args.benchmark
            )
            if ok:
                success_count += 1

    print("\n" + "=" * 76)
    print(f"📊 Ringkasan: Berhasil mengekspor {success_count}/{total_count} model ke TensorRT Engine.")
    print("   Pipeline inspeksi konveyor sekarang akan otomatis memuat file .engine ini!")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
