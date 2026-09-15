# =========================================================================
# CONFIG PRESETS FILTER OTOMATIS (Hasil Benchmark Preprocessing)
# Generated at: 2026-09-15 13:00:59
# =========================================================================

# Juara 1: 8. Adaptive Contrast Normalizer
# Skor Kualitas: 100.0/100 | Kehalusan: 100.0 | Riak: 0
FILTER_CONFIG_OPTIMAL = {'filter_type': 'adaptive_contrast', 'local_kernel': (21, 5), 'thresh': 30, 'morph': (11, 5)}

# Preset Berdasarkan Environment:
PRESET_FRONT_NORMAL = {
    "filter_type": "sobel",
    "clahe_clip": 2.5,
    "bilateral_d": 7,
    "thresh": 35,
    "morph": (13, 5),
    "smooth_method": "median_convolve"
}

PRESET_FRONT_DARK_OILY = {
    "filter_type": "morph_blackhat",
    "clahe_clip": 4.0,
    "blackhat_kernel": (35, 7),
    "thresh": 25,
    "morph": (15, 5),
    "smooth_method": "savgol_poly",
    "poly_order": 2,
    "window_length": 21
}

PRESET_SIDE_PERSPECTIVE = {
    "use_perspective_unwarp": True,
    "filter_type": "sobel_directional_y",
    "clahe_clip": 3.0,
    "bilateral_d": 7,
    "thresh": 28,
    "morph": (15, 3),
    "smooth_method": "median_convolve"
}


if __name__ == "__main__":
    import sys
    import os
    import subprocess

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 76)
    print("  [INFO] FILTER CONFIG PRESETS - Slat & Rib Conveyor AI")
    print("=" * 76)
    print("  File ini adalah KAMUS PRESET KONFIGURASI parameter filter.")
    print("  Digunakan sebagai modul konfigurasi untuk:")
    print("    - AI_Slat_Rib_Conveyor_V4_7_Robust_Profile_W_db_OK.py")
    print("    - api.py")
    print("-" * 76)
    print("  [PRESET AKTIF]")
    print(f"  * Juara 1 Optimal     : {FILTER_CONFIG_OPTIMAL}")
    print(f"  * Front View Normal   : {PRESET_FRONT_NORMAL}")
    print(f"  * Front View Gelap/Oli: {PRESET_FRONT_DARK_OILY}")
    print(f"  * Side View Miring    : {PRESET_SIDE_PERSPECTIVE}")
    print("=" * 76)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bench_script = os.path.join(script_dir, "Benchmark_Filter_Slat_Rib.py")

    print("\n[PETUNJUK] Untuk menjalankan PENGUJIAN VISUAL & BENCHMARK INTERAKTIF:")
    print(f"   Jalankan: python \"{bench_script}\"\n")

    try:
        ans = input(">> Buka modul Benchmark_Filter_Slat_Rib sekarang? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = "n"

    if ans in ["", "y", "ya", "yes"]:
        print("\n>> Membuka Benchmark Filter Slat & Rib...\n")
        subprocess.run([sys.executable, bench_script])
