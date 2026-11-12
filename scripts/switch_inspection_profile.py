#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PENGATUR PROFIL PARAMETER INSPEKSI TOP (SLAT & RIB CONVEYOR)
================================================================================
Program untuk memulihkan atau beralih antara:
1. Profil DEFAULT STABIL (Awal Pabrik / Golden Reference Baseline)
2. Profil HASIL KALIBRASI (Hasil Kalibrasi Otomatis Kamera [K])

Dapat dijalankan langsung melalui CLI:
  python switch_inspection_profile.py --restore-default
  python switch_inspection_profile.py --use-calibrated
  python switch_inspection_profile.py --status
Atau mode interaktif menu dengan menjalankan:
  python switch_inspection_profile.py
================================================================================
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Pastikan output terminal mendukung UTF-8 pada Windows CMD / PowerShell
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(CURRENT_DIR, "Vision_Engine_Core", "02_Top_Inspection_Slat_Rib", "configs")
ACTIVE_CONFIG_PATH = os.path.join(CONFIG_DIR, "top_inspection_config.json")
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, "top_inspection_config.default.json")
CALIBRATED_CONFIG_PATH = os.path.join(CONFIG_DIR, "top_inspection_config.calibrated.json")


def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[!] Gagal membaca {os.path.basename(path)}: {e}")
        return None


def save_json(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"[!] Gagal menyimpan ke {os.path.basename(path)}: {e}")
        return False


def get_profile_name(cfg):
    if not cfg:
        return "UNKNOWN / TIDAK TERBACA"
    return cfg.get("profile_name", "UNKNOWN")


def restore_default_profile():
    print("\n" + "=" * 76)
    print("🔄 PROSES PEMULIHAN PARAMETER DEFAULT STABIL...")
    print("=" * 76)

    default_cfg = load_json(DEFAULT_CONFIG_PATH)
    if not default_cfg:
        print(f"[ERROR] File profil default tidak ditemukan di:\n  {DEFAULT_CONFIG_PATH}")
        return False

    default_cfg["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ok = save_json(ACTIVE_CONFIG_PATH, default_cfg)

    if ok:
        th = default_cfg.get("thresholds", {})
        print("✅ BERHASIL DIKEMBALIKAN KE PROFIL DEFAULT STABIL!")
        print(f"[*] File Konfigurasi Aktif : {os.path.relpath(ACTIVE_CONFIG_PATH, CURRENT_DIR)}")
        print(f"[*] Profil Terpasang       : {default_cfg.get('profile_name', 'DEFAULT_STABIL')}")
        print(f"[*] Batas Toleransi Rib    : PV={th.get('rib_pv_threshold')}px, Tip={th.get('rib_tip_threshold')}px, Sag={th.get('rib_sagitta_threshold')}px")
        print(f"[*] Zona Prime Evaluasi    : [{th.get('rib_prime_min_y')}, {th.get('rib_prime_max_y')}] px")
        print(f"[*] Toleransi Void Plat    : Max={th.get('max_center_void')} px")
        print(f"[*] Kemiringan Bracket     : {th.get('bracket_tilt_deg')} derajat")
        print("=" * 76)
        print("ℹ️  Semua program inspeksi yang dijalankan berikutnya akan menggunakan parameter ini.")
        return True
    else:
        print("[ERROR] Gagal menulis ke file konfigurasi aktif.")
        return False


def use_calibrated_profile():
    print("\n" + "=" * 76)
    print("🔄 BERALIH KE PARAMETER HASIL KALIBRASI...")
    print("=" * 76)

    calib_cfg = load_json(CALIBRATED_CONFIG_PATH)
    if not calib_cfg:
        print(f"[!] File profil kalibrasi belum ditemukan di:\n  {CALIBRATED_CONFIG_PATH}")
        print("    ℹ️  Tips: Jalankan program inspeksi dan tekan tombol [K] untuk membuat profil kalibrasi.")
        return False

    calib_cfg["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ok = save_json(ACTIVE_CONFIG_PATH, calib_cfg)

    if ok:
        th = calib_cfg.get("thresholds", {})
        print("✅ BERHASIL BERALIH KE PROFIL HASIL KALIBRASI!")
        print(f"[*] File Konfigurasi Aktif : {os.path.relpath(ACTIVE_CONFIG_PATH, CURRENT_DIR)}")
        print(f"[*] Profil Terpasang       : {calib_cfg.get('profile_name', 'HASIL_KALIBRASI')}")
        print(f"[*] Batas Toleransi Rib    : PV={th.get('rib_pv_threshold')}px, Tip={th.get('rib_tip_threshold')}px, Sag={th.get('rib_sagitta_threshold')}px")
        print(f"[*] Zona Prime Evaluasi    : [{th.get('rib_prime_min_y')}, {th.get('rib_prime_max_y')}] px")
        print(f"[*] Toleransi Void Plat    : Max={th.get('max_center_void')} px")
        print(f"[*] Kemiringan Bracket     : {th.get('bracket_tilt_deg')} derajat")
        print("=" * 76)
        return True
    else:
        print("[ERROR] Gagal menulis ke file konfigurasi aktif.")
        return False


def show_status():
    print("\n" + "=" * 80)
    print("📋 STATUS PROFIL PARAMETER INSPEKSI (CAM-01 / CAM-03)")
    print("=" * 80)

    act_cfg = load_json(ACTIVE_CONFIG_PATH)
    def_cfg = load_json(DEFAULT_CONFIG_PATH)
    cal_cfg = load_json(CALIBRATED_CONFIG_PATH)

    act_name = get_profile_name(act_cfg)
    act_th = act_cfg.get("thresholds", {}) if act_cfg else {}
    def_th = def_cfg.get("thresholds", {}) if def_cfg else {}
    cal_th = cal_cfg.get("thresholds", {}) if cal_cfg else {}

    print(f"[*] STATUS PROFIL AKTIF SAAT INI : >>> {act_name} <<<")
    print(f"[*] Terakhir Diperbarui          : {act_cfg.get('updated_at', '-') if act_cfg else '-'}")
    print("-" * 80)
    print(f"{'PARAMETER METROLOGI':<26} | {'AKTIF':<15} | {'DEFAULT STABIL':<16} | {'HASIL KALIBRASI':<16}")
    print("-" * 80)

    keys = [
        ("rib_pv_threshold", "Rib Flatness PV (px)"),
        ("rib_tip_threshold", "Rib Tip Bending (px)"),
        ("rib_sagitta_threshold", "Rib Sagitta Parabola"),
        ("rib_dev_threshold", "Rib Deviasi Lokal"),
        ("max_center_void", "Max Void Plat Slat"),
        ("bracket_tilt_deg", "Kemiringan Tilt (deg)"),
        ("rib_prime_min_y", "Prime Zone Y-Min"),
        ("rib_prime_max_y", "Prime Zone Y-Max")
    ]

    for k, label in keys:
        v_act = str(act_th.get(k, "-"))
        v_def = str(def_th.get(k, "-"))
        v_cal = str(cal_th.get(k, "-"))
        print(f"{label:<26} | {v_act:<15} | {v_def:<16} | {v_cal:<16}")

    has_act_rect = "AKTIF (Affine)" if (act_cfg and act_cfg.get("rectification_matrix")) else "NON-AKTIF"
    has_def_rect = "AKTIF (Affine)" if (def_cfg and def_cfg.get("rectification_matrix")) else "NON-AKTIF"
    has_cal_rect = "AKTIF (Affine)" if (cal_cfg and cal_cfg.get("rectification_matrix")) else "NON-AKTIF"
    print(f"{'Virtual Rectification':<26} | {has_act_rect:<15} | {has_def_rect:<16} | {has_cal_rect:<16}")

    print("=" * 80 + "\n")


def interactive_menu():
    while True:
        act_cfg = load_json(ACTIVE_CONFIG_PATH)
        act_name = get_profile_name(act_cfg)

        print("\n" + "=" * 74)
        print("  ⚙️  MENU PENGELOLA PROFIL PARAMETER INSPEKSI KONVEYOR")
        print("=" * 74)
        print(f"  [Status Profil Aktif]: >>> {act_name} <<<")
        print("-" * 74)
        print("  [1] Kembalikan ke Parameter DEFAULT STABIL (Awal Pabrik)")
        print("  [2] Beralih ke Parameter HASIL KALIBRASI (Terbaru dari Tombol K)")
        print("  [3] Lihat Tabel Perbandingan Parameter Lengkap")
        print("  [0] Keluar")
        print("=" * 74)

        try:
            choice = input("  Masukkan pilihan [0-3]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break

        if choice == "1":
            restore_default_profile()
        elif choice == "2":
            use_calibrated_profile()
        elif choice == "3":
            show_status()
        elif choice in ("0", "q", "Q", "exit"):
            print("\n[*] Selesai.\n")
            break
        else:
            print("[!] Pilihan tidak valid, silakan masukkan 0, 1, 2, atau 3.")


def main():
    parser = argparse.ArgumentParser(
        description="Pengatur Profil Parameter Inspeksi Top Slat & Rib Conveyor",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--restore-default", "-r", action="store_true",
                       help="Kembalikan file konfigurasi ke Profil Default Stabil (Baseline Terbukti)")
    group.add_argument("--use-calibrated", "-c", action="store_true",
                       help="Beralih ke Profil Hasil Kalibrasi Otomatis Kamera (Tombol K)")
    group.add_argument("--status", "-s", action="store_true",
                       help="Tampilkan status profil aktif dan perbandingan parameter")

    args = parser.parse_args()

    if args.restore_default:
        success = restore_default_profile()
        sys.exit(0 if success else 1)
    elif args.use_calibrated:
        success = use_calibrated_profile()
        sys.exit(0 if success else 1)
    elif args.status:
        show_status()
        sys.exit(0)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
