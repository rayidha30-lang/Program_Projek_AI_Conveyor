"""
==============================================================================
JETSON CUDA 12.6 & NVIDIA TENSOR ENVIRONMENT BOOTSTRAPPER
==============================================================================
Modul ini bertugas memastikan seluruh pustaka dinamis (dynamic shared libraries)
CUDA 12.6, cuDNN 9, cuSPARSELt, cuPTI, cuBLAS, dan NVTX pada NVIDIA Jetson Orin Nano
termuat ke dalam proses memori global sebelum PyTorch / Ultralytics YOLO diimpor.

Dengan modul ini:
- Script dapat dijalankan langsung via IDE (Run button), python3, maupun bash
  tanpa mengalami `ImportError: libcupti.so.12` atau `libcusparseLt.so.0`.
- Memastikan akselerasi CUDA GPU selalu aktif untuk inferensi YOLO.
==============================================================================
"""

import os
import sys
import ctypes
import warnings

# Filter peringatan non-kritis (ABI torchvision C++ ops & half deprecation)
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", message=".*half.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

def bootstrap_cuda_environment():
    """
    Memuat pustaka bersama CUDA & Tensor NVIDIA ke namespace RTLD_GLOBAL
    sesuai urutan dependensi dependensi library PyTorch JetPack 6.
    """
    # 1. Update LD_LIBRARY_PATH environment
    lib_dirs = [
        "/home/ai/.local/cuda-12.6/usr/local/cuda-12.6/targets/aarch64-linux/lib",
        "/home/ai/.local/cuda-12.6/usr/lib/aarch64-linux-gnu",
        "/home/ai/.local/lib/python3.10/site-packages/nvidia/cuda_cupti/lib",
        "/home/ai/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib",
        "/home/ai/.local/lib/python3.10/site-packages/nvidia/nvtx/lib",
        "/usr/lib/aarch64-linux-gnu",
    ]
    cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
    to_add = [d for d in lib_dirs if os.path.exists(d) and d not in cur_ld]
    if to_add:
        os.environ["LD_LIBRARY_PATH"] = ":".join(to_add + ([cur_ld] if cur_ld else []))

    # 2. Muat library spesifik ke RTLD_GLOBAL sesuai topologi dependensi
    target_libs = [
        "/home/ai/.local/cuda-12.6/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so.12",
        "/home/ai/.local/cuda-12.6/usr/lib/aarch64-linux-gnu/libcudnn.so.9",
        "/home/ai/.local/lib/python3.10/site-packages/nvidia/nvtx/lib/libnvToolsExt.so.1",
        "/home/ai/.local/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12",
        "/home/ai/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib/libcusparseLt.so.0",
        "/home/ai/.local/cuda-12.6/usr/local/cuda-12.6/targets/aarch64-linux/lib/libnvJitLink.so.12",
        "/home/ai/.local/cuda-12.6/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcurand.so.10",
        "/home/ai/.local/cuda-12.6/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcufft.so.11",
        "/home/ai/.local/cuda-12.6/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcusparse.so.12",
        "/home/ai/.local/cuda-12.6/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcublasLt.so.12",
        "/home/ai/.local/cuda-12.6/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcublas.so.12",
    ]

    loaded_count = 0
    for p in target_libs:
        if os.path.exists(p):
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                loaded_count += 1
            except Exception:
                pass

    return loaded_count

# Jalankan bootstrap saat modul diimpor
_LOADED_SO_COUNT = bootstrap_cuda_environment()
