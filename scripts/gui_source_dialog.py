#!/usr/bin/env python3
"""
========================================================================================
🚀 STANDALONE CROSS-PLATFORM GUI DIALOG RUNNER (ZERO-CRASH ISOLATION)
========================================================================================
Deskripsi:
  Dialog mandiri terisolasi untuk pemilihan Sumber Kamera & File Video (CAM-01 Top & CAM-02 Side).
  Didesain 100% bebas ketergantungan OpenCV / CUDA / PyTorch untuk mencegah konflik memori
  X11 / GTK (Anti "stack smashing detected") pada Jetson Orin Nano (Ubuntu Linux) dan Windows.

Fitur:
  1. Linux (Jetson Ubuntu): Menggunakan Native GNOME/Ubuntu Zenity sebagai prioritas utama.
     - Mode Dual: Menggunakan zenity --forms untuk menampilkan panel CAM-01 Top dan CAM-02 Side
       secara bersamaan dalam SATU jendela dialog GTK terpadu.
     - 100% kebal dari buffer overflow / glibc stack smashing Tcl/Tk.
  2. Windows: Menggunakan Tkinter murni dengan desain antarmuka modern (Segoe UI).
  3. Protokol Output Ganda: Selalu mencetak RESULT_SOURCE dan RESULT_FILE agar pembaruan
     sumber video selalu berhasil diterapkan seketika pada runner OpenCV.
========================================================================================
"""

import os
import sys
import glob
import shutil
import subprocess
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VIDEO_DIR = os.path.join(PROJECT_ROOT, "Video_Testing")


def get_available_cameras(max_devices=6):
    """Mendeteksi index webcam fisik USB yang aktif secara cepat dan aman."""
    cams = []
    if sys.platform.startswith("win"):
        try:
            import cv2
            for i in range(min(max_devices, 5)):
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if cap is not None and cap.isOpened():
                        cams.append(i)
                        cap.release()
                except Exception:
                    pass
        except Exception:
            pass
        if not cams:
            cams = [0, 1, 2, 3]
    else:
        nodes = sorted(glob.glob("/dev/video[0-9]*"))
        for node in nodes:
            try:
                idx = int(os.path.basename(node).replace("video", ""))
                if idx < max_devices and idx not in cams:
                    cams.append(idx)
            except ValueError:
                pass
        if not cams:
            cams = [0, 1, 2, 3]
    return cams


def has_zenity():
    """Mengecek apakah utilitas native GNOME/Ubuntu 'zenity' tersedia di Linux."""
    return sys.platform.startswith("linux") and shutil.which("zenity") is not None


# ======================================================================================
# STRATEGI 1: ZENITY RUNNER (NATIVE UBUNTU LINUX GTK - 100% ZERO CRASH & UNIFIED FORMS)
# ======================================================================================

def _pick_zenity_file(title="Pilih File Video"):
    """Helper untuk membuka file picker Zenity."""
    cmd_f = [
        "zenity",
        "--file-selection",
        f"--title={title}",
        f"--filename={VIDEO_DIR}/",
        "--file-filter=File Video | *.mp4 *.avi *.mkv *.mov *.wmv",
        "--file-filter=Semua File | *"
    ]
    try:
        rf = subprocess.run(cmd_f, stdout=subprocess.PIPE, text=True)
        if rf.returncode == 0 and rf.stdout.strip():
            sel = rf.stdout.strip()
            if os.path.exists(sel):
                return os.path.abspath(sel)
    except Exception:
        pass
    return None


def run_zenity_file(title="Pilih File Video", initial_dir=None):
    """Membuka file dialog native Ubuntu GNOME via Zenity."""
    init_d = initial_dir if (initial_dir and os.path.exists(initial_dir)) else VIDEO_DIR
    cmd = [
        "zenity",
        "--file-selection",
        f"--title={title}",
        f"--filename={init_d}/",
        "--file-filter=File Video | *.mp4 *.avi *.mkv *.mov *.wmv",
        "--file-filter=Semua File | *"
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            selected = res.stdout.strip()
            if selected and os.path.exists(selected):
                norm_p = os.path.abspath(selected)
                print(f"RESULT_FILE:{norm_p}")
                print(f"RESULT_SOURCE:{norm_p}")
                return True
    except Exception as e:
        sys.stderr.write(f"[Zenity] Error file dialog: {e}\n")
    return False


def run_zenity_source(title="Pilih Sumber Kamera / Video", default_source=None):
    """Membuka dialog visual pemilihan input kamera/video via Zenity."""
    cams = get_available_cameras()
    is_default_cam = str(default_source).strip().isdigit() if default_source else False
    def_cam_idx = int(default_source) if is_default_cam else (cams[0] if cams else 0)

    cmd = [
        "zenity",
        "--list",
        "--radiolist",
        f"--title={title}",
        "--text=Pilih salah satu sumber input pengujian:",
        "--column=Pilih",
        "--column=ID / Tipe",
        "--column=Keterangan",
        "--width=520",
        "--height=380"
    ]

    for c in cams:
        is_sel = "TRUE" if (is_default_cam and c == def_cam_idx) else ("FALSE" if is_default_cam else ("TRUE" if c == cams[0] else "FALSE"))
        cmd.extend([is_sel, f"CAM_{c}", f"Live USB Webcam #{c}"])

    file_label = f"File Video: {os.path.basename(str(default_source))}" if (default_source and not is_default_cam) else "Pilih File Rekaman Video Baru..."
    cmd.extend(["FALSE", "FILE", file_label])

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            choice = res.stdout.strip()
            if choice.startswith("CAM_"):
                cam_id = choice.replace("CAM_", "")
                print(f"RESULT_SOURCE:{cam_id}")
                print(f"RESULT_FILE:{cam_id}")
                return True
            elif choice == "FILE":
                sel = _pick_zenity_file(title=title)
                if sel:
                    print(f"RESULT_SOURCE:{sel}")
                    print(f"RESULT_FILE:{sel}")
                    return True
    except Exception as e:
        sys.stderr.write(f"[Zenity] Error source dialog: {e}\n")
    return False


def run_zenity_dual(default_top=None, default_side=None, side_id="CAM-02"):
    """
    Membuka pengaturan dual camera terpadu menggunakan Zenity --forms.
    Menampilkan pilihan CAM-01 Top dan CAM-02 Side dalam SATU dialog GTK bersamaan (bukan satu-satu).
    """
    cams = get_available_cameras()

    # Opsi dropdown untuk CAM-01 Top
    top_def_label = f"Tetap ({os.path.basename(str(default_top)) if default_top else 'Default'})"
    top_options = [top_def_label]
    for c in cams:
        top_options.append(f"Webcam #{c}")
    top_options.append("Pilih File Video Baru...")
    top_combo_str = "|".join(top_options)

    # Opsi dropdown untuk CAM-02 Side
    side_def_label = f"Tetap ({os.path.basename(str(default_side)) if default_side else 'Default'})"
    side_options = [side_def_label]
    for c in cams:
        side_options.append(f"Webcam #{c}")
    side_options.append("Pilih File Video Baru...")
    side_combo_str = "|".join(side_options)

    cmd = [
        "zenity",
        "--forms",
        f"--title=🎥 Pengaturan Sumber Dual Kamera (CAM-01 Top & {side_id} Side)",
        "--text=Atur sumber input untuk CAM-01 Top dan CAM-02 Side secara bersamaan:",
        "--separator=:::",
        f"--add-combo=[CAM-01 TOP] Plat & Sirip Rib",
        f"--combo-values={top_combo_str}",
        f"--add-combo=[{side_id} SIDE] Baut & Pin",
        f"--combo-values={side_combo_str}",
        "--width=560"
    ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            output = res.stdout.strip()
            parts = output.split(":::")
            ch_top = parts[0].strip() if len(parts) > 0 else ""
            ch_side = parts[1].strip() if len(parts) > 1 else ""

            # 1. Resolusi Sumber CAM-01 Top
            final_top = str(default_top) if default_top else "0"
            if ch_top.startswith("Webcam #"):
                final_top = ch_top.replace("Webcam #", "").strip()
            elif ch_top == "Pilih File Video Baru...":
                f_top = _pick_zenity_file(title="Pilih File Video [CAM-01 TOP]")
                if f_top:
                    final_top = f_top
            elif ch_top == top_def_label:
                final_top = str(default_top) if default_top else "0"

            # 2. Resolusi Sumber Side
            final_side = str(default_side) if default_side else "1"
            if ch_side.startswith("Webcam #"):
                final_side = ch_side.replace("Webcam #", "").strip()
            elif ch_side == "Pilih File Video Baru...":
                f_side = _pick_zenity_file(title=f"Pilih File Video [{side_id} SIDE]")
                if f_side:
                    final_side = f_side
            elif ch_side == side_def_label:
                final_side = str(default_side) if default_side else "1"

            if final_top:
                print(f"RESULT_DUAL_TOP:{final_top}")
            if final_side:
                print(f"RESULT_DUAL_SIDE:{final_side}")
            return True
    except Exception as e:
        sys.stderr.write(f"[Zenity Dual] Error: {e}\n")
    return False


# ======================================================================================
# STRATEGI 2: TKINTER RUNNER (STANDALONE UNTUK WINDOWS & LINUX NON-ZENITY)
# ======================================================================================

def run_tkinter_file(title="Pilih File Video", initial_dir=None):
    """Membuka dialog file explorer menggunakan Tkinter murni."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    if sys.platform.startswith("win"):
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass

    init_d = initial_dir if (initial_dir and os.path.exists(initial_dir)) else VIDEO_DIR
    fp = filedialog.askopenfilename(
        parent=root,
        title=title,
        initialdir=init_d,
        filetypes=[
            ("File Video (*.mp4;*.avi;*.mkv;*.mov;*.wmv)", "*.mp4 *.avi *.mkv *.mov *.wmv"),
            ("Semua File (*.*)", "*.*")
        ]
    )
    try:
        root.destroy()
    except Exception:
        pass

    if fp and os.path.exists(fp):
        norm_p = os.path.abspath(fp)
        print(f"RESULT_FILE:{norm_p}")
        print(f"RESULT_SOURCE:{norm_p}")
        return True
    return False


def run_tkinter_source(title="Pilih Sumber Kamera / Video", default_source=None):
    """Dialog pemilihan sumber kamera/video menggunakan Tkinter."""
    import tkinter as tk
    from tkinter import filedialog

    cams = get_available_cameras()
    root = tk.Tk()
    root.title(title)

    if sys.platform.startswith("win"):
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass

    root.geometry("500x420")
    root.minsize(450, 380)

    f_fam = "Segoe UI" if sys.platform.startswith("win") else "DejaVu Sans"
    f_title = (f_fam, 11, "bold")
    f_sec = (f_fam, 10, "bold")
    f_reg = (f_fam, 9)

    selected_result = {"val": default_source}

    lbl_info = tk.Label(root, text=f"🎥 {title}", font=f_title, pady=10)
    lbl_info.pack()

    frame_options = tk.LabelFrame(root, text="Pilihan Sumber Input", font=f_sec, padx=15, pady=10)
    frame_options.pack(fill="both", expand=True, padx=15, pady=5)

    is_default_cam = str(default_source).strip().isdigit() if default_source else False
    var_choice = tk.StringVar(master=root, value="webcam" if is_default_cam else "file")
    default_cam_idx = int(default_source) if is_default_cam else (cams[0] if cams else 0)
    var_cam_idx = tk.IntVar(master=root, value=default_cam_idx)

    root.var_choice = var_choice
    root.var_cam_idx = var_cam_idx

    rb_cam = tk.Radiobutton(frame_options, text="Gunakan Perangkat Live Webcam USB:", variable=var_choice, value="webcam", font=(f_fam, 9, "bold"))
    rb_cam.pack(anchor="w", pady=(0, 4))

    frame_cams = tk.Frame(frame_options, padx=20)
    frame_cams.pack(fill="x", pady=2)
    candidate_cams = sorted(list(set(cams + [0, 1, 2, 3])))
    for c_idx in candidate_cams:
        tag = " (Aktif)" if c_idx in cams else ""
        tk.Radiobutton(frame_cams, text=f"Webcam #{c_idx}{tag}", variable=var_cam_idx, value=c_idx, command=lambda: var_choice.set("webcam"), font=f_reg).pack(anchor="w", pady=2)

    rb_file = tk.Radiobutton(frame_options, text="Gunakan File Rekaman Video:", variable=var_choice, value="file", font=(f_fam, 9, "bold"))
    rb_file.pack(anchor="w", pady=(12, 4))

    lbl_file_path = tk.Label(
        frame_options,
        text=os.path.basename(str(default_source)) if (default_source and not is_default_cam) else "(Belum memilih file)",
        font=(f_fam, 8, "italic"),
        fg="#555555"
    )
    lbl_file_path.pack(anchor="w", padx=20)

    file_selected_path = {"path": str(default_source) if (default_source and not is_default_cam) else None}

    def on_browse():
        var_choice.set("file")
        f = filedialog.askopenfilename(
            parent=root,
            title="Pilih File Video Pengujian",
            initialdir=VIDEO_DIR,
            filetypes=[
                ("File Video (*.mp4;*.avi;*.mkv;*.mov;*.wmv)", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                ("Semua File (*.*)", "*.*")
            ]
        )
        if f:
            file_selected_path["path"] = os.path.abspath(f)
            lbl_file_path.config(text=os.path.basename(f), fg="blue")

    btn_browse = tk.Button(frame_options, text="📁 Buka File Explorer...", command=on_browse, font=f_reg)
    btn_browse.pack(anchor="w", padx=20, pady=4)

    def on_ok():
        if var_choice.get() == "webcam":
            selected_result["val"] = str(var_cam_idx.get())
        else:
            if file_selected_path["path"]:
                selected_result["val"] = str(file_selected_path["path"])
            else:
                selected_result["val"] = default_source
        root.destroy()

    frame_btn = tk.Frame(root, pady=10)
    frame_btn.pack(side="bottom", fill="x", padx=15)
    tk.Button(frame_btn, text="Pilih Sumber Ini", bg="#007acc", fg="white", font=(f_fam, 9, "bold"), width=15, command=on_ok).pack(side="right", padx=5)
    tk.Button(frame_btn, text="Batal", font=f_reg, width=10, command=root.destroy).pack(side="right")

    root.mainloop()
    if selected_result["val"]:
        val_str = str(selected_result["val"])
        print(f"RESULT_SOURCE:{val_str}")
        print(f"RESULT_FILE:{val_str}")
        return True
    return False


def run_tkinter_dual(default_top=None, default_side=None, side_id="CAM-02"):
    """
    Dialog visual pengaturan dual camera terpadu (CAM-01 Top & CAM-02 Side dalam 1 jendela).
    Tampilan bawaan untuk Windows.
    """
    import tkinter as tk
    from tkinter import filedialog

    cams = get_available_cameras()
    root = tk.Tk()
    root.title(f"Pengaturan Dual Kamera (CAM-01 Top & {side_id} Side)")

    if sys.platform.startswith("win"):
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass

    root.geometry("620x540")
    root.minsize(560, 480)

    f_fam = "Segoe UI" if sys.platform.startswith("win") else "DejaVu Sans"
    f_title = (f_fam, 11, "bold")
    f_sec = (f_fam, 10, "bold")
    f_reg = (f_fam, 9)
    f_sm = (f_fam, 8)

    res = {"top": default_top, "side": default_side}

    lbl_t = tk.Label(root, text="🎥 PENGATURAN SUMBER DUAL KAMERA", font=f_title, pady=8)
    lbl_t.pack()
    lbl_sub = tk.Label(
        root,
        text=f"Webcam terdeteksi: {len(cams)} perangkat ({', '.join(['#'+str(c) for c in cams]) if cams else 'Tidak ada'})",
        font=(f_fam, 8, "italic"),
        fg="#444444"
    )
    lbl_sub.pack(pady=(0, 8))

    # --- PANEL TOP ---
    f_top = tk.LabelFrame(root, text="[CAM-01 TOP] Deteksi Plat Slat & Bilah Rib", font=f_sec, padx=12, pady=6)
    f_top.pack(fill="x", padx=15, pady=5)

    is_top_cam = str(default_top).strip().isdigit() if default_top else False
    var_top_choice = tk.StringVar(master=root, value="webcam" if (is_top_cam or (cams and not default_top)) else "file")
    var_top_cam = tk.IntVar(master=root, value=int(default_top) if is_top_cam else (cams[0] if cams else 0))
    top_file_path = {"path": str(default_top) if (default_top and not is_top_cam) else None}

    candidate_cams = sorted(list(set(cams + [0, 1, 2, 3])))
    f_top_cams = tk.Frame(f_top)
    f_top_cams.pack(fill="x")
    tk.Radiobutton(f_top_cams, text="Webcam:", variable=var_top_choice, value="webcam", font=(f_fam, 9, "bold")).pack(side="left")
    for c_idx in candidate_cams:
        tk.Radiobutton(f_top_cams, text=f"Cam #{c_idx}", variable=var_top_cam, value=c_idx, command=lambda: var_top_choice.set("webcam"), font=f_reg).pack(side="left", padx=6)

    f_top_file = tk.Frame(f_top, pady=4)
    f_top_file.pack(fill="x")
    tk.Radiobutton(f_top_file, text="File Video:", variable=var_top_choice, value="file", font=(f_fam, 9, "bold")).pack(side="left")
    lbl_top_fname = tk.Label(f_top_file, text=os.path.basename(str(top_file_path["path"])) if top_file_path["path"] else "(Default Video)", font=f_sm, fg="#0055aa")
    lbl_top_fname.pack(side="left", padx=6)

    def browse_top():
        var_top_choice.set("file")
        f = filedialog.askopenfilename(parent=root, title="Pilih Video CAM-01 Top", initialdir=VIDEO_DIR)
        if f:
            top_file_path["path"] = os.path.abspath(f)
            lbl_top_fname.config(text=os.path.basename(f), fg="blue")

    tk.Button(f_top_file, text="Pilih...", command=browse_top, font=f_sm).pack(side="right")

    # --- PANEL SIDE ---
    f_side = tk.LabelFrame(root, text=f"[{side_id} SIDE] Deteksi Baut, Pin & Bracket Lekukan", font=f_sec, padx=12, pady=6)
    f_side.pack(fill="x", padx=15, pady=8)

    is_side_cam = str(default_side).strip().isdigit() if default_side else False
    var_side_choice = tk.StringVar(master=root, value="webcam" if (is_side_cam or (len(cams) >= 2 and not default_side)) else "file")
    default_side_idx = int(default_side) if is_side_cam else (cams[1] if len(cams) >= 2 else (cams[0] if cams else 0))
    var_side_cam = tk.IntVar(master=root, value=default_side_idx)
    side_file_path = {"path": str(default_side) if (default_side and not is_side_cam) else None}

    f_side_cams = tk.Frame(f_side)
    f_side_cams.pack(fill="x")
    tk.Radiobutton(f_side_cams, text="Webcam:", variable=var_side_choice, value="webcam", font=(f_fam, 9, "bold")).pack(side="left")
    for c_idx in candidate_cams:
        tk.Radiobutton(f_side_cams, text=f"Cam #{c_idx}", variable=var_side_cam, value=c_idx, command=lambda: var_side_choice.set("webcam"), font=f_reg).pack(side="left", padx=6)

    f_side_file = tk.Frame(f_side, pady=4)
    f_side_file.pack(fill="x")
    tk.Radiobutton(f_side_file, text="File Video:", variable=var_side_choice, value="file", font=(f_fam, 9, "bold")).pack(side="left")
    lbl_side_fname = tk.Label(f_side_file, text=os.path.basename(str(side_file_path["path"])) if side_file_path["path"] else "(Default Video)", font=f_sm, fg="#0055aa")
    lbl_side_fname.pack(side="left", padx=6)

    def browse_side():
        var_side_choice.set("file")
        f = filedialog.askopenfilename(parent=root, title=f"Pilih Video {side_id} Side", initialdir=VIDEO_DIR)
        if f:
            side_file_path["path"] = os.path.abspath(f)
            lbl_side_fname.config(text=os.path.basename(f), fg="blue")

    tk.Button(f_side_file, text="Pilih...", command=browse_side, font=f_sm).pack(side="right")

    root.var_top_choice = var_top_choice
    root.var_top_cam = var_top_cam
    root.var_side_choice = var_side_choice
    root.var_side_cam = var_side_cam

    def on_confirm_all():
        if var_top_choice.get() == "webcam":
            res["top"] = str(var_top_cam.get())
        elif top_file_path["path"]:
            res["top"] = str(top_file_path["path"])

        if var_side_choice.get() == "webcam":
            res["side"] = str(var_side_cam.get())
        elif side_file_path["path"]:
            res["side"] = str(side_file_path["path"])

        root.destroy()

    f_bottom = tk.Frame(root, pady=15)
    f_bottom.pack(side="bottom", fill="x", padx=15)
    tk.Button(f_bottom, text="Terapkan & Mulai", bg="#007acc", fg="white", font=(f_fam, 10, "bold"), width=18, command=on_confirm_all).pack(side="right", padx=5)
    tk.Button(f_bottom, text="Gunakan Bawaan", font=f_reg, width=15, command=root.destroy).pack(side="right")

    root.mainloop()
    if res.get("top"):
        print(f"RESULT_DUAL_TOP:{res['top']}")
    if res.get("side"):
        print(f"RESULT_DUAL_SIDE:{res['side']}")
    return True


# ======================================================================================
# ENTRY POINT
# ======================================================================================

def main():
    parser = argparse.ArgumentParser(description="Standalone Cross-Platform GUI Dialog Runner")
    parser.add_argument("--dialog", type=str, required=True, choices=["file", "source", "dual"])
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--initial-dir", type=str, default="")
    parser.add_argument("--default", type=str, default="")
    parser.add_argument("--top", type=str, default="")
    parser.add_argument("--side", type=str, default="")
    parser.add_argument("--side-id", type=str, default="CAM-02")
    args = parser.parse_args()

    # Prioritas Linux: Gunakan Zenity 100% native GTK untuk garansi kebal stack smashing Tkinter
    if sys.platform.startswith("linux") and has_zenity():
        if args.dialog == "dual":
            ok = run_zenity_dual(default_top=args.top, default_side=args.side, side_id=args.side_id)
            if ok:
                sys.exit(0)
        elif args.dialog == "source":
            ok = run_zenity_source(title=args.title or "Pilih Sumber Kamera / Video", default_source=args.default)
            if ok:
                sys.exit(0)
        elif args.dialog == "file":
            ok = run_zenity_file(title=args.title or "Pilih File Video", initial_dir=args.initial_dir)
            if ok:
                sys.exit(0)

    # Windows / Fallback Non-Zenity: Gunakan Tkinter
    if args.dialog == "dual":
        ok = run_tkinter_dual(default_top=args.top, default_side=args.side, side_id=args.side_id)
        sys.exit(0 if ok else 1)
    elif args.dialog == "source":
        ok = run_tkinter_source(title=args.title or "Pilih Sumber Kamera / Video", default_source=args.default)
        sys.exit(0 if ok else 1)
    elif args.dialog == "file":
        ok = run_tkinter_file(title=args.title or "Pilih File Video", initial_dir=args.initial_dir)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
