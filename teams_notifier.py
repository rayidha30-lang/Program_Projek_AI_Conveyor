"""
==============================================================================
NODE-RED & MS TEAMS NOTIFIER (NON-BLOCKING ASYNC SENDER)
Target: AI Slat & Rib Conveyor Inspection System
==============================================================================
Deskripsi:
  Modul pengirim notifikasi hasil inspeksi (terutama insiden NG / Abnormality)
  ke Node-RED lokal (http://localhost:1880/ai-inspection) yang kemudian
  diteruskan ke Microsoft Teams melalui Power Automate Adaptive Card.

Fitur Utama:
  1. Zero Frame-Drop (Non-Blocking):
     Pengiriman HTTP dilakukan di background worker thread melalui queue.Queue,
     sehingga tidak mengurangi FPS deteksi kamera sama sekali.
  2. Smart Deduplication & Cooldown:
     Mencegah spamming notifikasi multi-frame saat benda NG bergerak di konveyor.
  3. Skema Payload Komprehensif:
     Menyediakan nilai default aman (CAM-01, CAM-02, CAM-03) sehingga
     Adaptive Card di Node-RED tidak pernah error jika salah satu kamera standby.
==============================================================================
"""

import os
import sys
import time
import json
import queue
import threading
import urllib.request
import urllib.error
from datetime import datetime

# Faktor skala kalibrasi fisik conveyor (Ground Truth: Rib 750mm / 1457px & Plat Slat 820mm / 1592px = 0.515 mm/px)
SCALE_FACTOR_MM_PER_PX = 0.515


class NodeRedTeamsNotifier:
    """
    Klien asinkron non-blocking untuk memicu notifikasi MS Teams via Node-RED.
    """

    def __init__(
        self,
        endpoint_url="http://127.0.0.1:1880/ai-inspection",
        enabled=True,
        cooldown_sec=3.0,
        timeout=3.0
    ):
        self.endpoint_url = endpoint_url
        self.enabled = bool(enabled)
        self.cooldown_sec = float(cooldown_sec)
        self.timeout = float(timeout)

        # Anti-spam tracker (Timestamp kirim terakhir & riwayat key)
        self._last_send_time = 0.0
        self._sent_keys = set()
        self._lock = threading.Lock()

        # Non-blocking Queue & Worker Thread
        self._queue = queue.Queue(maxsize=50)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="TeamsNotifierWorker"
        )
        self._worker_thread.start()

        status = "ENABLED" if self.enabled else "DISABLED"
        print(f"[*] NodeRedTeamsNotifier aktif ({status}) -> {self.endpoint_url}")

    def is_enabled(self):
        return self.enabled

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def reset_history(self):
        with self._lock:
            self._sent_keys.clear()
            self._last_send_time = 0.0

    def send_raw_payload(self, payload: dict, dedup_key=None, force=False):
        """
        Memasukkan payload ke antrean kirim jika memenuhi syarat anti-spam.
        """
        if not self.enabled:
            return False

        now = time.time()
        with self._lock:
            if not force:
                # Cek deduplikasi key (misal ID slat)
                if dedup_key and dedup_key in self._sent_keys:
                    return False

                # Cek cooldown waktu
                if (now - self._last_send_time) < self.cooldown_sec:
                    return False

            if dedup_key:
                self._sent_keys.add(dedup_key)
                # Batasi ukuran history agar hemat memori
                if len(self._sent_keys) > 200:
                    self._sent_keys.clear()

            self._last_send_time = now

        try:
            self._queue.put_nowait(payload)
            return True
        except queue.Full:
            print("[TEAMS WARN] Antrean notifikasi penuh! Payload dilewati.")
            return False

    def send_top_inspection_ng(
        self,
        slat_res=None,
        rib_res=None,
        slat_number=None,
        rib_number=None,
        inspection_id=None,
        evidence_url=None,
        image_url=None,
        cam02_res=None,
        cam03_res=None,
        force=False
    ):
        """
        Format khusus untuk sistem kamera inspeksi tampak atas (CAM-01 / CAM-03).
        Mengekstrak metrik Flatness Slat & Rib secara dinamis ke dalam daftar defects.
        Jika hanya 1 komponen yang NG, payload hanya berisi komponen tersebut.
        Jika beberapa komponen NG sekaligus, disatukan dalam 1 payload terintegrasi.
        """
        slat_res = slat_res or {}
        rib_res = rib_res or {}

        is_slat_ng = bool(slat_res.get("is_ng", False))
        is_rib_ng = bool(rib_res.get("is_ng", False))

        now_dt = datetime.now()
        ts_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

        # Tentukan ID inspeksi (Prioritaskan Traceability ID dari database jika ada)
        slat_tag = f"SLAT-{slat_number}" if slat_number is not None else "SLAT"
        rib_tag = f"RIB-{rib_number}" if rib_number is not None else "RIB"
        if inspection_id:
            insp_id = str(inspection_id)
        else:
            insp_id = f"INSP-{slat_tag}-{rib_tag}-{now_dt.strftime('%H%M%S')}"

        # Dedup key unik per objek fisik
        dedup_key = f"{slat_tag}_{rib_tag}" if (slat_number or rib_number) else None

        # Nilai metrik CAM-01 dasar
        slat_pv = float(slat_res.get("pv_flatness", slat_res.get("pv", 0.0)) or 0.0)
        rib_dev = float(rib_res.get("max_dev", rib_res.get("dev", 0.0)) or 0.0)
        max_dev = max(slat_pv, rib_dev)

        # ======================================================================
        # DAFTAR CACAT DINAMIS (HANYA KOMPONEN YANG BENAR-BENAR BERSTATUS NG)
        # ======================================================================
        defects = []

        # 1. Evaluasi Bilah Sirip (Rib - CAM-01)
        if is_rib_ng:
            rib_msg = rib_res.get("message", "Rib Bending / Deformasi")
            r_dev = float(rib_res.get("inner_dev", rib_res.get("dev", rib_dev)))
            r_tip = float(rib_res.get("tip_dev", 0.0))
            r_pv = float(rib_res.get("pv_flatness", 0.0))
            r_sag = float(rib_res.get("sagitta", 0.0))

            scale = SCALE_FACTOR_MM_PER_PX
            detail_parts = []
            if r_dev > 0:
                detail_parts.append(f"Dev Badan: {r_dev * scale:.1f}mm ({r_dev:.1f}px)")
            if r_tip > 0:
                detail_parts.append(f"Dev Tip: {r_tip * scale:.1f}mm ({r_tip:.1f}px)")
            if r_pv > 0:
                detail_parts.append(f"PV: {r_pv * scale:.1f}mm ({r_pv:.1f}px)")
            if r_sag > 0:
                detail_parts.append(f"Sagitta: {r_sag * scale:.1f}mm ({r_sag:.1f}px)")
            detail_str = ", ".join(detail_parts) if detail_parts else f"Deviasi: {rib_dev * scale:.1f}mm ({rib_dev:.1f}px)"

            defects.append({
                "camera": "CAM-01",
                "component": "Bilah Sirip (CAM-01)",
                "component_short": "Sirip Rib",
                "defect_type": "Bending / Deformasi Sirip" if ("bending" in rib_msg.lower() or not rib_msg) else rib_msg,
                "metrics": detail_str,
                "severity": "CRITICAL"
            })

        # 2. Evaluasi Plat Konveyor (Slat - CAM-01)
        if is_slat_ng:
            slat_eval = slat_res.get("eval_mode", "")
            s_pv = float(slat_res.get("pv_flatness", slat_res.get("pv", slat_pv)))
            s_sag = float(slat_res.get("sagitta", 0.0))
            s_void = int(slat_res.get("center_void", 0))

            if "void" in slat_eval.lower() or "mangap" in slat_eval.lower() or s_void > 1450:
                d_type = "Celah Sambungan Mangap / Renggang"
            elif "sag" in slat_eval.lower() or "bengkok" in slat_eval.lower() or "melengkung" in slat_eval.lower():
                d_type = "Plat Melengkung (Sagitta Curvature)"
            elif "pv" in slat_eval.lower() or "kasar" in slat_eval.lower():
                d_type = "Kerataan Celah Kasar (Flatness PV Deflection)"
            else:
                d_type = slat_eval if slat_eval else "Deformasi Plat Slat"

            scale = SCALE_FACTOR_MM_PER_PX
            detail_parts = []
            if s_pv > 0:
                detail_parts.append(f"Flatness PV: {s_pv * scale:.1f}mm ({s_pv:.1f}px)")
            if s_sag > 0:
                detail_parts.append(f"Sagitta: {s_sag * scale:.1f}mm ({s_sag:.1f}px)")
            if s_void > 0:
                detail_parts.append(f"Center Void: {s_void}px")
            detail_str = ", ".join(detail_parts) if detail_parts else f"PV: {slat_pv * scale:.1f}mm ({slat_pv:.1f}px)"

            defects.append({
                "camera": "CAM-01",
                "component": "Plat Konveyor (CAM-01)",
                "component_short": "Plat Slat",
                "defect_type": d_type,
                "metrics": detail_str,
                "severity": "CRITICAL" if (s_void > 2500 or s_pv > 25) else "MEDIUM"
            })

        # 3. Evaluasi Kamera Samping Kanan (CAM-02: Bolt & Pin)
        if cam02_res:
            bolt2 = str(cam02_res.get("bolt", "")).upper()
            pin2 = str(cam02_res.get("pin", "")).upper()
            if "NG" in bolt2:
                defects.append({
                    "camera": "CAM-02",
                    "component": "Baut Sisi Kanan (CAM-02)",
                    "component_short": "Bolt RH",
                    "defect_type": "Baut Kendur / Hilang (Bolt Abnormality)",
                    "metrics": bolt2,
                    "severity": "HIGH"
                })
            if "NG" in pin2:
                defects.append({
                    "camera": "CAM-02",
                    "component": "Pin Sisi Kanan (CAM-02)",
                    "component_short": "Pin RH",
                    "defect_type": "Pin Rantai Miring / Tergeser (Misalignment)",
                    "metrics": pin2,
                    "severity": "HIGH"
                })

        # 4. Evaluasi Kamera Samping Kiri (CAM-03: Bolt & Pin)
        if cam03_res:
            bolt3 = str(cam03_res.get("bolt", "")).upper()
            pin3 = str(cam03_res.get("pin", "")).upper()
            if "NG" in bolt3:
                defects.append({
                    "camera": "CAM-03",
                    "component": "Baut Sisi Kiri (CAM-03)",
                    "component_short": "Bolt LH",
                    "defect_type": "Baut Kendur / Hilang (Bolt Abnormality)",
                    "metrics": bolt3,
                    "severity": "HIGH"
                })
            if "NG" in pin3:
                defects.append({
                    "camera": "CAM-03",
                    "component": "Pin Sisi Kiri (CAM-03)",
                    "component_short": "Pin LH",
                    "defect_type": "Pin Rantai Miring / Tergeser (Misalignment)",
                    "metrics": pin3,
                    "severity": "HIGH"
                })

        # Hanya kirim jika terbukti ada komponen yang NG (atau jika force=True untuk pengujian)
        if not defects and not force:
            return False

        # Judul kartu seragam untuk seluruh jenis cacat (single maupun multi-abnormalitas)
        if defects or force:
            summary_title = "🚨 AI CAMERA CONVEYOR INSPECTION"
        else:
            summary_title = "INSPEKSI RUTIN (STATUS NORMAL)"

        cam01_data = {
            "slat": "BENDING (NG)" if is_slat_ng else "NORMAL (OK)",
            "rib": "BENDING (NG)" if is_rib_ng else "NORMAL (OK)",
            "pv_deviation": round(slat_pv, 2),
            "max_deviation": round(max_dev, 2),
            "pv_deviation_mm": round(slat_pv * SCALE_FACTOR_MM_PER_PX, 2),
            "max_deviation_mm": round(max_dev * SCALE_FACTOR_MM_PER_PX, 2)
        }

        # Data Kamera CAM-02 & CAM-03 (Fallback kompatibilitas)
        cam02_data = cam02_res or {
            "bolt": "NORMAL (OK)",
            "pin": "NORMAL (OK)"
        }
        cam03_data = cam03_res or {
            "bolt": "NORMAL (OK)",
            "pin": "NORMAL (OK)"
        }

        # URL Tautan Dashboard & Foto
        if not evidence_url and insp_id:
            evidence_url = f"http://localhost:5173/?trace={insp_id}"

        # Sematkan direct_url per komponen spesifik kamera (Pendekatan B: Deep-Linking)
        base_trace_url = evidence_url or f"http://localhost:5173/?trace={insp_id}"
        for d in defects:
            cam = d.get("camera", "CAM-01")
            d["direct_url"] = f"{base_trace_url}&cam={cam}"

        payload = {
            "inspection_id": insp_id,
            "overall_result": "NG" if (defects or not force) else "OK",
            "timestamp": ts_str,
            "evidence_url": evidence_url,
            "image_url": image_url,
            "defect_count": len(defects),
            "summary_title": summary_title,
            "defects": defects,
            # Kompatibilitas ke belakang (Legacy fields)
            "cam01": cam01_data,
            "cam02": cam02_data,
            "cam03": cam03_data
        }

        return self.send_raw_payload(payload, dedup_key=dedup_key, force=force)

    def send_side_inspection_ng(
        self,
        camera_id="CAM-01",
        side="RIGHT",
        components=None,
        bracket_details=None,
        inspection_id=None,
        evidence_url=None,
        image_url=None,
        force=False
    ):
        """
        Format khusus untuk inspeksi samping rantai konveyor (Baut, Pin & Lekukan Bracket).
        Menyusun daftar defects dinamis untuk Adaptive Card MS Teams.
        """
        components = components or []
        now_dt = datetime.now()
        ts_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

        if not inspection_id:
            inspection_id = f"INSP-SIDE-{camera_id}-{now_dt.strftime('%H%M%S')}"

        defects = []
        for comp in components:
            name = str(comp.get("name", "")).upper()
            res = str(comp.get("result", "")).upper()
            if res == "NG":
                if name == "BOLT":
                    defects.append({
                        "camera": camera_id,
                        "component": f"Baut Rantai Konveyor ({camera_id} {side})",
                        "component_short": f"Bolt {side}",
                        "defect_type": "Baut Kendor / Terlepas (Loose/Missing Bolt)",
                        "metrics": "Status: ABNORMAL (NG)",
                        "severity": "HIGH"
                    })
                elif name == "PIN":
                    defects.append({
                        "camera": camera_id,
                        "component": f"Pin Rantai Konveyor ({camera_id} {side})",
                        "component_short": f"Pin {side}",
                        "defect_type": "Pin Rantai Bergeser / Miring (Misalignment)",
                        "metrics": "Status: ABNORMAL (NG)",
                        "severity": "HIGH"
                    })
                elif name == "LEKUKAN":
                    score = bracket_details.get("anomaly_score", 0.0) if bracket_details else 0.0
                    defects.append({
                        "camera": camera_id,
                        "component": f"Bracket Plat Slat ({camera_id} {side})",
                        "component_short": f"Bracket {side}",
                        "defect_type": "Retak / Kerusakan Bracket Plat (PatchCore Crack Anomaly)",
                        "metrics": f"Score: {score:.3f} (Batas: 0.850)",
                        "severity": "CRITICAL"
                    })

        if not defects and not force:
            return False

        base_trace_url = evidence_url or f"http://localhost:5173/?trace={inspection_id}"
        for d in defects:
            d["direct_url"] = f"{base_trace_url}&cam={camera_id}"

        payload = {
            "inspection_id": inspection_id,
            "overall_result": "NG",
            "timestamp": ts_str,
            "evidence_url": base_trace_url,
            "image_url": image_url,
            "defect_count": len(defects),
            "summary_title": "🚨 AI CAMERA CONVEYOR INSPECTION",
            "defects": defects,
            "camera_id": camera_id,
            "side": side,
            "cam01": {
                "slat": "NORMAL (OK)",
                "rib": "NORMAL (OK)",
                "pv_deviation": 0.0,
                "max_deviation": 0.0,
                "pv_deviation_mm": 0.0,
                "max_deviation_mm": 0.0
            },
            "cam02": {
                "bolt": "NG (Baut Kendor)" if any(d.get("component_short") == f"Bolt {side}" for d in defects) else "NORMAL (OK)",
                "pin": "NG (Pin Geser)" if any(d.get("component_short") == f"Pin {side}" for d in defects) else "NORMAL (OK)"
            },
            "cam03": {
                "bolt": "NORMAL (OK)",
                "pin": "NORMAL (OK)"
            }
        }
        dedup_key = f"{camera_id}_{side}_{len(defects)}"
        return self.send_raw_payload(payload, dedup_key=dedup_key, force=force)

    def _worker_loop(self):
        """Worker thread untuk transmisi HTTP tanpa memblokir thread OpenCV."""
        while not self._stop_event.is_set():
            try:
                payload = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._dispatch_http(payload)
            except Exception as e:
                print(f"[TEAMS ERROR] Gagal mengirim webhook ke Node-RED: {e}")
            finally:
                self._queue.task_done()

    def _dispatch_http(self, payload):
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint_url,
            data=data_bytes,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            resp_body = resp.read().decode("utf-8", errors="replace")
            print(f"[TEAMS OK] Notifikasi terkirim | HTTP {resp.status} | ID: {payload.get('inspection_id')}")

    def close(self):
        """Hentikan worker thread dengan aman."""
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)


# ==============================================================================
# CLI Testing Entrypoint
# ==============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Node-RED Teams Notifier")
    parser.add_argument("--url", default="http://127.0.0.1:1880/ai-inspection", help="Node-RED endpoint")
    parser.add_argument(
        "--scenario",
        choices=["rib-only", "slat-only", "side-only", "multi-top", "multi-all", "ok"],
        default="rib-only",
        help="Skenario pengujian notifikasi Teams"
    )
    parser.add_argument("--slat-no", type=int, default=183, help="Nomor objek slat/rib")
    args = parser.parse_args()

    print(f"[*] Menjalankan uji coba notifikasi ke {args.url} (Skenario: {args.scenario})...")
    notifier = NodeRedTeamsNotifier(endpoint_url=args.url)

    if args.scenario == "rib-only":
        print("[*] Mengirim test: HANYA Rib NG (Bilah Sirip CAM-01)...")
        success = notifier.send_top_inspection_ng(
            slat_res={"is_ng": False, "pv_flatness": 5.2, "center_void": 450},
            rib_res={"is_ng": True, "max_dev": 4.8, "sagitta": 4.2, "message": "Bilah Sirip Bengkok (Rib Deflection)"},
            slat_number=args.slat_no,
            force=True
        )
    elif args.scenario == "slat-only":
        print("[*] Mengirim test: HANYA Plat Konveyor NG (Plat Slat CAM-01 Celah Renggang)...")
        success = notifier.send_top_inspection_ng(
            slat_res={"is_ng": True, "pv_flatness": 18.5, "center_void": 1959, "eval_mode": "Celah Sambungan Renggang (Void Mangap)"},
            rib_res={"is_ng": False, "max_dev": 1.4, "sagitta": 1.1},
            slat_number=args.slat_no,
            force=True
        )
    elif args.scenario == "side-only":
        print("[*] Mengirim test: HANYA Side Inspection NG (Baut Kendor & Pin Geser CAM-01)...")
        success = notifier.send_side_inspection_ng(
            camera_id="CAM-01",
            side="RIGHT",
            components=[
                {"name": "BOLT", "result": "NG"},
                {"name": "PIN", "result": "NG"},
                {"name": "LEKUKAN", "result": "OK"}
            ],
            force=True
        )
    elif args.scenario == "multi-top":
        print("[*] Mengirim test: MULTI ABNORMALITAS (Rib + Plat Slat CAM-01)...")
        success = notifier.send_top_inspection_ng(
            slat_res={"is_ng": True, "pv_flatness": 19.8, "center_void": 2100, "eval_mode": "Celah Sambungan Renggang (Void Mangap)"},
            rib_res={"is_ng": True, "max_dev": 5.4, "sagitta": 4.9, "message": "Bilah Sirip Bengkok & Patah"},
            slat_number=args.slat_no,
            force=True
        )
    elif args.scenario == "multi-all":
        print("[*] Mengirim test: MULTI ABNORMALITAS GABUNGAN (CAM-01 Rib+Slat, CAM-02 Pin, CAM-03 Bolt)...")
        success = notifier.send_top_inspection_ng(
            slat_res={"is_ng": True, "pv_flatness": 18.5, "center_void": 1959, "eval_mode": "Celah Renggang"},
            rib_res={"is_ng": True, "max_dev": 4.8, "sagitta": 4.2, "message": "Bilah Sirip Bengkok"},
            cam02_res={"bolt": "OK", "pin": "NG (Pin Geser 12mm)"},
            cam03_res={"bolt": "NG (Baut Lepas)", "pin": "OK"},
            slat_number=args.slat_no,
            force=True
        )
    else:  # ok
        print("[*] Mengirim test: Status NORMAL (OK)...")
        payload = {
            "inspection_id": f"INSP-OK-{int(time.time())}",
            "overall_result": "OK",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        success = notifier.send_raw_payload(payload, force=True)

    print(f"[*] Pengiriman diantrekan: {success}")
    # Tunggu worker menyelesaikan pengiriman HTTP
    time.sleep(2.0)
    notifier.close()
    print("[*] Selesai.")
