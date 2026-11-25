"""Generate printable high-contrast ArUco 4x4 fiducial marker sheet (IDs 0..3) for stationary conveyor rails."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT.parent / "Documentation" / "aruco_rail_markers_4x4_sheet.png"

MARKER_SPECS = (
    (0, "ID 0 : TOP-LEFT RAIL (Rangka Kiri Atas)"),
    (1, "ID 1 : TOP-RIGHT RAIL (Rangka Kanan Atas)"),
    (2, "ID 2 : BOT-RIGHT RAIL (Rangka Kanan Bawah)"),
    (3, "ID 3 : BOT-LEFT RAIL (Rangka Kiri Bawah)"),
)


def generate_aruco_sheet(out_path: str | Path = DEFAULT_OUT, marker_px: int = 520) -> Path:
    """Create a 2x2 A4-proportioned printable sheet (2480 x 3508 px @ 300 DPI) with quiet zones & cut guides."""
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("Modul cv2.aruco tidak ditemukan di instalasi OpenCV saat ini.")

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    sheet_w, sheet_h = 2480, 3508  # A4 @ 300 DPI
    canvas = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)

    # Header
    cv2.putText(
        canvas,
        "PLAT FIDUCIAL ARUCO DICT_4X4_50 — AI SLAT & RIB CONVEYOR",
        (140, 140),
        cv2.FONT_HERSHEY_DUPLEX,
        1.7,
        (20, 20, 20),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Tempel / baut pada RANGKA DIAM kiri-kanan konveyor (di luar jalur gerak sabuk & ingot). Pertahankan border putih!",
        (140, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (60, 60, 60),
        2,
        cv2.LINE_AA,
    )
    cv2.line(canvas, (120, 260), (sheet_w - 120, 260), (40, 40, 40), 3)

    cell_w = 980
    cell_h = 1320
    origins = [
        (180, 360),
        (1320, 360),
        (1320, 1820),
        (180, 1820),
    ]

    for (m_id, label), (ox, oy) in zip(MARKER_SPECS, origins):
        # Dashed/solid cut border
        cv2.rectangle(canvas, (ox, oy), (ox + cell_w, oy + cell_h), (120, 120, 120), 3)
        # Quiet zone white pad + marker centered
        if hasattr(cv2.aruco, "generateImageMarker"):
            m_gray = cv2.aruco.generateImageMarker(aruco_dict, m_id, marker_px)
        else:
            m_gray = np.zeros((marker_px, marker_px), dtype=np.uint8)
            cv2.aruco.drawMarker(aruco_dict, m_id, marker_px, m_gray, 1)
        m_bgr = cv2.cvtColor(m_gray, cv2.COLOR_GRAY2BGR)

        mx = ox + (cell_w - marker_px) // 2
        my = oy + 140
        # Outer mounting frame with screw-hole crosshairs
        pad = 90
        cv2.rectangle(canvas, (mx - pad, my - pad), (mx + marker_px + pad, my + marker_px + pad), (0, 0, 0), 4)
        for hx, hy in (
            (mx - pad + 28, my - pad + 28),
            (mx + marker_px + pad - 28, my - pad + 28),
            (mx - pad + 28, my + marker_px + pad - 28),
            (mx + marker_px + pad - 28, my + marker_px + pad - 28),
        ):
            cv2.circle(canvas, (hx, hy), 14, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.drawMarker(canvas, (hx, hy), (0, 0, 0), cv2.MARKER_CROSS, 20, 2)

        canvas[my : my + marker_px, mx : mx + marker_px] = m_bgr
        cv2.putText(
            canvas,
            label,
            (ox + 45, my + marker_px + pad + 95),
            cv2.FONT_HERSHEY_DUPLEX,
            0.95,
            (10, 10, 10),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Ukuran cetak disarankan: {int(round(marker_px / 11.81))} - 80 mm persegi (Laser-etched / Acrylic Doff)",
            (ox + 45, my + marker_px + pad + 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (80, 80, 80),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        "Catatan: CameraPoseWatchdog otomatis beralih dari mode GPU Rail Trapezoid ke mode ARUCO_FIDUCIAL begitu >= 2 marker terdeteksi.",
        (140, sheet_h - 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.88,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )

    target = Path(out_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), canvas):
        raise RuntimeError(f"Gagal menulis file gambar ArUco ke {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate printable ArUco 4x4 sheet for conveyor rail calibration.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output PNG path")
    parser.add_argument("--marker-px", type=int, default=520, help="Marker square size in pixels (@300 DPI)")
    args = parser.parse_args()

    saved = generate_aruco_sheet(out_path=args.out, marker_px=args.marker_px)
    print(f"[SUCCESS] ArUco sheet generated: {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
