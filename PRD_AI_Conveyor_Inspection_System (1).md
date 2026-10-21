# PRODUCT REQUIREMENTS DOCUMENT (PRD)
# AI Conveyor Inspection System

**Project:** AI Slat Rib Conveyor  
**Document:** Product Requirements Document  
**Version:** V1.0  
**Status:** Development / Prototype  
**Frontend:** React + Vite + HTML/CSS  
**Communication:** REST API  
**Database:** Microsoft SQL Server (SSMS)  
**Vision System:** Computer Vision / AI  
**Camera:** 3 IP CCTV Cameras

---

## 1. Project Overview

AI Conveyor Inspection System merupakan sistem inspeksi berbasis computer vision yang digunakan untuk melakukan monitoring kondisi komponen conveyor secara otomatis menggunakan tiga kamera.

Sistem dirancang untuk mendeteksi:

1. Keberadaan bolt/baut dan pin serta keretakan pada lengkungan ujung sisi conveyor (opsional).
3. Deformasi rib/sirip conveyor.
4. Deformasi plate/slat conveyor.

Hasil inspeksi dari sistem vision dikirim melalui REST API dan disimpan pada Microsoft SQL Server. Informasi inspeksi kemudian ditampilkan pada web dashboard menggunakan React + Vite.

Metode computer vision/AI belum dikunci dan akan ditentukan berdasarkan hasil eksperimen, karakteristik objek, kondisi kamera, serta hasil validasi.

---

## 2. Background / Problem Statement

Conveyor merupakan salah satu komponen penting dalam proses produksi. Kondisi komponen conveyor perlu dipantau untuk memastikan komponen tetap berada dalam kondisi yang sesuai.

Abnormality yang menjadi fokus proyek ini meliputi:

- Bolt/baut hilang.
- Pin hilang atau terlepas.
- Keretakan pada sisi ujung conveyor (opsional).
- Rib/sirip mengalami deformasi.
- Plate/slat mengalami deformasi.

Jika abnormality tidak diketahui secara cepat, kondisi tersebut berpotensi berlanjut dan menyebabkan gangguan pada conveyor atau proses produksi.

Inspeksi manual juga memiliki keterbatasan karena membutuhkan pengamatan operator secara berkala dan hasilnya belum tentu terdokumentasi secara digital.

Oleh karena itu, diperlukan sistem inspeksi otomatis berbasis computer vision yang mampu memberikan informasi abnormality secara lebih cepat, konsisten, dan terdokumentasi.

---

## 3. Project Objectives

### 3.1 Primary Objectives

1. Mendeteksi keberadaan bolt/baut pada sisi kanan dan kiri conveyor.
2. Mendeteksi keberadaan dan kondisi pin pada sisi kanan dan kiri conveyor.
3. Mendeteksi kondisi ujung sisi kanan dan kiri conveyor retak/copot tetapi opsional.
4. Mendeteksi deformasi rib/sirip conveyor.
5. Mendeteksi deformasi plate/slat conveyor.
6. Menghasilkan status inspeksi OK, NG, atau INVALID.
7. Menampilkan hasil inspeksi pada web dashboard.
8. Menyimpan hasil inspeksi ke SQL Server.
9. Menyimpan informasi abnormality dan bukti visual jika tersedia.

### 3.2 Secondary Objectives

1. Mempercepat informasi mengenai abnormality conveyor.
2. Mengurangi ketergantungan terhadap inspeksi visual manual.
3. Meningkatkan traceability hasil inspeksi.
4. Menyediakan histori abnormality untuk kebutuhan monitoring dan maintenance.
5. Menjadi dasar pengembangan sistem monitoring conveyor yang lebih terintegrasi.

---

## 4. Scope of Project

### 4.1 In Scope

- Akuisisi image/video dari tiga kamera.
- Pemrosesan image menggunakan computer vision/AI.
- Deteksi bolt sisi kanan.
- Deteksi pin sisi kanan.
- Deteksi bolt sisi kiri.
- Deteksi pin sisi kiri.
- Deteksi kondisi ujung sisi kanan dan kiri conveyor.
- Deteksi deformasi rib/sirip.
- Deteksi deformasi plate/slat.
- Penentuan status OK/NG/INVALID.
- REST API.
- SQL Server database.
- React + Vite web dashboard.
- Monitoring hasil inspeksi.
- Inspection history.
- Abnormality image storage.

### 4.2 Out of Scope

Untuk tahap awal sistem tidak mencakup:

- Perbaikan conveyor secara otomatis.
- Penggantian bolt/pin secara otomatis.
- Kontrol langsung terhadap mesin conveyor.
- Predictive maintenance berbasis lifetime komponen.
- Integrasi PLC/MES/SCADA apabila belum diperlukan.
- Pengukuran metrologi presisi dalam satuan mm, kecuali ditambahkan sebagai requirement pada tahap berikutnya.

---

## 5. Inspection Point

Sistem memiliki enam inspection point utama dari tiga kamera:

| Camera | Side / Area | Inspection Point |
|---|---|---|
| CAM-01 | Right Side | Bolt Right |
| CAM-01 | Right Side | Pin Right |
| CAM-01 | Right Side | lekukan Right |
| CAM-02 | Left Side | Bolt Left |
| CAM-02 | Left Side | Pin Left |
| CAM-02 | Left Side | lekukan Left |
| CAM-03 | Top / Surface | Rib / Sirip |
| CAM-03 | Top / Surface | Plate / Slat |

CAM-01 dan CAM-02 mempunyai fungsi inspeksi yang sama, tetapi masing-masing digunakan untuk sisi conveyor yang berbeda.

---

## 6. Camera Configuration

### 6.1 CAM-01 — Right Side Inspection

CAM-01 digunakan untuk melakukan inspeksi komponen pada sisi kanan conveyor:

- Bolt Right
- Pin Right
- Kondisi lekukan Right

Requirement:

- Sistem dapat mendeteksi keberadaan bolt.
- Sistem dapat mendeteksi keberadaan pin.
- Sistem dapat mendeteksi kondisi lekukan ujung sisi conveyor.
- Sistem dapat mendeteksi pin yang bergeser/terlepas jika karakteristik visual memungkinkan.
- Sistem memberikan status untuk setiap komponen.

### 6.2 CAM-02 — Left Side Inspection

CAM-02 digunakan untuk melakukan inspeksi komponen pada sisi kiri conveyor:

- Bolt Left
- Pin Left
- Kondisi lekukan left

Requirement:

- Sistem dapat mendeteksi keberadaan bolt.
- Sistem dapat mendeteksi keberadaan pin.
- Sistem dapat mendeteksi kondisi lekukan ujung sisi conveyor.
- Sistem dapat mendeteksi pin yang bergeser/terlepas jika karakteristik visual memungkinkan.
- Sistem memberikan status untuk setiap komponen.

### 6.3 CAM-03 — Top / Surface Inspection

CAM-03 digunakan untuk melakukan inspeksi pada permukaan conveyor:

- Rib / Sirip
- Plate / Slat

Fokus inspeksi:

- Deformasi rib.
- Deformasi plate.
- Penyimpangan bentuk/geometri dari kondisi referensi.

---

## 7. Inspection Logic

### 7.1 Bolt Inspection

Kondisi yang harus dapat dibedakan:

| Condition | Result |
|---|---|
| Bolt tersedia pada posisi normal | OK |
| Bolt tidak ditemukan | NG |
| Image/detection tidak valid | INVALID |

### 7.2 Pin Inspection

Kondisi yang harus dapat dibedakan:

| Condition | Result |
|---|---|
| Pin tersedia pada posisi normal | OK |
| Pin tidak ditemukan | NG |
| Pin terlepas/bergeser dari posisi normal | NG |
| Image/detection tidak valid | INVALID |

### Tambahkan untuk mendeteksi ujung sisi kanan dan kiri lekukan conveyor

### 7.3 Rib Inspection

Sistem melakukan evaluasi bentuk/geometri rib terhadap kondisi referensi.

Potential parameters:

- Posisi.
- Kontur.
- Sudut.
- Straightness.
- Profile deviation.
- Geometric deviation.

Status akhir:

- OK: memenuhi kriteria.
- NG: ditemukan deformasi.
- INVALID: inspeksi tidak dapat dilakukan secara valid.

### 7.4 Plate Inspection

Sistem melakukan evaluasi kondisi plate/slat terhadap kondisi referensi.

Potential parameters:

- Surface profile.
- Edge/profile.
- Flatness.
- Deviation.
- Geometric distortion.

Threshold harus ditentukan berdasarkan data sampel OK dan NG serta validasi lapangan.

---

## 8. Inspection Status

Sistem menggunakan tiga status:

### OK

Komponen memenuhi kriteria inspeksi.

### NG

Ditemukan abnormality yang memenuhi kriteria NG.

### INVALID

Sistem tidak dapat melakukan inspeksi secara valid.

Contoh penyebab:

- Kamera offline.
- Image blur.
- Lighting tidak sesuai.
- Object tertutup.
- ROI tidak ditemukan.
- Confidence terlalu rendah.
- Geometry tidak dapat dihitung.
- Image tidak valid.

Status INVALID diperlukan agar sistem tidak memaksa kondisi yang tidak dapat diinspeksi menjadi OK atau NG.

---

## 9. Overall Result Logic

Overall result ditentukan dari seluruh inspection point.

Inspection point:

- Bolt Right
- Pin Right
- Lekukan Right 
- Bolt Left
- Pin Left
- Lekukan Left
- Rib
- Plate

Prioritas status:

**NG > INVALID > OK**

Contoh:

```text
Bolt Right      = OK
Pin Right       = OK
Lekukan Right   = OK
Bolt Left       = OK
Pin Left        = NG
Lekukan Left    = OK
Rib             = OK
Plate           = OK

Overall = NG
```

Jika tidak ada NG tetapi terdapat INVALID:

```text
Bolt Right = OK
Pin Right  = OK
Bolt Left  = OK
Pin Left   = INVALID
Rib        = OK
Plate      = OK

Overall = INVALID
```

---

## 10. System Architecture

Arsitektur sistem:

```text
CAM-01 ─┐
        │
CAM-02 ─┼──> AI Vision / Computer Vision
        │             │
CAM-03 ─┘             ↓
                 Inspection Logic
                       │
                       ↓
                    REST API
                       │
                       ↓
                 Backend Service
                       │
                       ↓
                  SQL Server
                       │
                       ↓
               React + Vite
                Web Dashboard
```

### 10.1 Image Acquisition

Kamera menyediakan image/video stream kepada AI Vision PC.

### 10.2 AI Vision

AI Vision melakukan:

1. Frame capture.
2. Image preprocessing.
3. Object detection / segmentation / anomaly detection / geometry analysis.
4. Inspection logic.
5. Result generation.
6. Abnormality image capture jika diperlukan.

### 10.3 REST API

REST API menjadi communication layer antara vision system, backend, database, dan frontend.

### 10.4 Database

SQL Server menyimpan data inspeksi dan histori.

### 10.5 Frontend

React + Vite digunakan untuk menampilkan hasil inspeksi dan monitoring.

---

## 11. Computer Vision / AI Strategy

Metode AI belum ditentukan secara final.

Metode yang dapat dievaluasi:

- Object Detection.
- Instance Segmentation.
- Image Classification.
- Anomaly Detection.
- Geometry-based Inspection.
- Edge / Contour Analysis.
- Hybrid Computer Vision + AI.

Rekomendasi awal berdasarkan jenis problem:

| Inspection | Candidate Method |
|---|---|
| Bolt Right/Left | Object Detection / Classification |
| Pin Right/Left | Object Detection / Segmentation |
| Rib | Segmentation / Geometry / Anomaly Detection |
| Plate | Geometry / Segmentation / Anomaly Detection |

Pemilihan metode harus berdasarkan eksperimen menggunakan data aktual dari kamera.

Sistem tidak mengharuskan satu metode yang sama digunakan untuk seluruh inspection point.

---

## 12. AI Model Evaluation

Evaluasi model dibagi menjadi dua kelompok.

### 12.1 Model Metrics

Untuk detection/AI:

- Precision.
- Recall.
- F1-score.
- mAP.
- Confidence distribution.

### 12.2 Inspection Metrics

Untuk aplikasi industrial:

- False OK.
- False NG.
- Miss Detection.
- Detection latency.
- Inspection repeatability.
- Invalid rate.

False OK merupakan metrik penting karena kondisi abnormal yang diklasifikasikan sebagai normal dapat menyebabkan abnormality tidak terdeteksi.

---

## 13. REST API Requirements

Contoh endpoint:

```text
POST /api/inspection
GET  /api/inspection/latest
GET  /api/inspection/history
GET  /api/inspection/{id}
GET  /api/statistics
GET  /api/abnormality
```

### POST /api/inspection

Digunakan AI Vision untuk mengirim hasil inspeksi.

Contoh payload:

```json
{
  "inspection_id": "000125",
  "inspection_time": "2026-09-18T08:30:15",
  "overall_result": "NG",
  "inspection_results": [
    {
      "camera_id": "CAM-01",
      "side": "RIGHT",
      "component": "BOLT",
      "result": "OK"
    },
    {
      "camera_id": "CAM-01",
      "side": "RIGHT",
      "component": "PIN",
      "result": "OK"
    },
    {
      "camera_id": "CAM-01",
      "side": "RIGHT",
      "component": "Lekukan",
      "result": "OK"
    },
    {
      "camera_id": "CAM-02",
      "side": "LEFT",
      "component": "BOLT",
      "result": "OK"
    },
    {
      "camera_id": "CAM-02",
      "side": "LEFT",
      "component": "PIN",
      "result": "NG"
    },
    {
      "camera_id": "CAM-02",
      "side": "LEFT",
      "component": "Lekukan",
      "result": "NG"
    },
    {
      "camera_id": "CAM-03",
      "component": "RIB",
      "result": "OK"
    },
    {
      "camera_id": "CAM-03",
      "component": "PLATE",
      "result": "OK"
    }
  ]
}
```

---

## 14. Database Requirements

Database menggunakan Microsoft SQL Server.

### 14.1 Inspection

Menyimpan informasi utama inspeksi.

```text
InspectionID
InspectionTime
OverallResult
TraceabilityCode
Shift
PIC
CreatedAt
```

### 14.2 InspectionResult

Menyimpan hasil setiap inspection point.

```text
InspectionResultID
InspectionID
CameraID
Side
Component
Result
Confidence
AbnormalityType
ImagePath
InspectionTime
```

### 14.3 Abnormality

Menyimpan detail abnormality.

```text
AbnormalityID
InspectionID
InspectionResultID
CameraID
Component
AbnormalityType
Severity
Description
ImagePath
CreatedAt
```

Struktur database dapat dikembangkan setelah requirement final dan API contract ditentukan.

---

## 15. Web Dashboard Requirements

Frontend menggunakan React + Vite.

### 15.1 Main Dashboard

Dashboard menampilkan kondisi conveyor terkini.

Informasi utama:

- Overall status.
- Status Bolt Right.
- Status Pin Right.
- Status Bolt Left.
- Status Pin Left.
- Status Rib.
- Status Plate.
- Last inspection time.
- Camera status.

Contoh layout:

```text
+------------------------------------------------------+
|              AI CONVEYOR INSPECTION                  |
+------------------------------------------------------+
| Overall Status: NG                                   |
+--------------------------+---------------------------+
| RIGHT SIDE - CAM-01      | LEFT SIDE - CAM-02       |
| Bolt : OK                | Bolt : OK                |
| Pin  : OK                | Lekukan : OK             |
| Lekukan : OK             | Pin  : NG                |
+--------------------------+---------------------------+
|              SURFACE - CAM-03                        |
| Rib   : OK                                           |
| Plate : OK                                           |
+------------------------------------------------------+
```

---

## 16. Camera Monitoring

Dashboard menampilkan visual dari:

- CAM-01 — Right Side.
- CAM-02 — Left Side.
- CAM-03 — Top/Surface.

Setiap kamera harus memiliki indikator:

- Online.
- Offline.
- Connection Error.

---

## 17. KPI Dashboard

KPI yang dapat ditampilkan:

- Total Inspection.
- Total OK.
- Total NG.
- Total INVALID.
- Bolt abnormality.
- Pin abnormality.
- Rib deformation.
- Plate deformation.
- Abnormality rate.
- Latest inspection.
- Camera availability.

Contoh:

```text
TOTAL INSPECTION       1,245
OK                     1,180
NG                        48
INVALID                   17
ABNORMALITY RATE        3.9%
```

---

## 18. Abnormality Monitoring

Ketika terjadi NG, dashboard menampilkan:

```text
ABNORMALITY DETECTED

Time       : 08:32:14
Camera     : CAM-02
Side       : LEFT
Component  : PIN
Condition  : PIN MISSING
Result     : NG

[Abnormal Image]
```

Informasi tersebut disimpan ke database.

---

## 19. Inspection History

Dashboard menyediakan histori inspeksi.

Contoh:

| Time | Camera | Side | Component | Result | Abnormality |
|---|---|---|---|---|---|
| 08:30:10 | CAM-01 | Right | Bolt | OK | - |
| 08:30:10 | CAM-01 | Right | Pin | OK | - |
| 08:30:10 | CAM-01 | Right | Lekukan | OK | - |
| 08:30:10 | CAM-02 | Left | Bolt | OK | - |
| 08:31:24 | CAM-02 | Left | Pin | NG | Missing |
| 08:31:24 | CAM-02 | Left | Lekukan | NG | Missing |
| 08:32:05 | CAM-03 | - | Rib | OK | - |
| 08:32:05 | CAM-03 | - | Plate | NG | Deformation |

Filter:

- Date.
- Time.
- Camera.
- Side.
- Component.
- Result.
- Abnormality type.

---

## 20. Abnormality Image Management

Untuk hasil NG, sistem menyimpan image abnormality jika memungkinkan.

Contoh struktur:

```text
/images
    /bolt
    /pin
    /rib
    /plate
```

Contoh filename:

```text
INS_000125_CAM02_LEFT_PIN_20260918_083214.jpg
```

Database menyimpan path image yang terkait dengan InspectionID.

---

## 21. Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-001 | Sistem dapat menerima stream CAM-01 | High |
| FR-002 | Sistem dapat menerima stream CAM-02 | High |
| FR-003 | Sistem dapat menerima stream CAM-03 | High |
| FR-004 | Sistem dapat mendeteksi Bolt Right | High |
| FR-005 | Sistem dapat mendeteksi Pin Right | High |
| FR-005 | Sistem dapat mendeteksi Lekukan Right | High |
| FR-006 | Sistem dapat mendeteksi Bolt Left | High |
| FR-007 | Sistem dapat mendeteksi Pin Left | High |
| FR-005 | Sistem dapat mendeteksi Lekukan Left | High |
| FR-008 | Sistem dapat mendeteksi deformasi Rib | High |
| FR-009 | Sistem dapat mendeteksi deformasi Plate | High |
| FR-010 | Sistem menghasilkan OK/NG/INVALID | High |
| FR-011 | Hasil dikirim melalui REST API | High |
| FR-012 | Hasil disimpan ke SQL Server | High |
| FR-013 | Dashboard menampilkan hasil terbaru | High |
| FR-014 | Dashboard menampilkan camera monitoring | High |
| FR-015 | Sistem menyimpan abnormality image | Medium |
| FR-016 | Dashboard menyediakan inspection history | Medium |
| FR-017 | Dashboard menyediakan filtering | Medium |
| FR-018 | Dashboard menampilkan statistik abnormality | Medium |

---

## 22. Non-Functional Requirements

### Performance

- Sistem mampu melakukan inspeksi secara near real-time.
- Latency detection ke dashboard harus rendah.
- Nilai FPS dan latency final ditentukan setelah hardware dan metode AI ditetapkan.

### Reliability

Sistem harus:

- Mendeteksi kamera offline.
- Mencatat communication error.
- Tidak kehilangan data inspeksi jika terjadi gangguan komunikasi sementara.
- Dapat melakukan reconnect terhadap stream kamera.
- Menyimpan error log.

### Traceability

Setiap hasil inspeksi harus memiliki:

- Inspection ID.
- Timestamp.
- Camera ID.
- Side.
- Component.
- Result.
- Image path jika tersedia.

### Security

REST API harus memiliki mekanisme autentikasi apabila sistem digunakan pada production network.

---

## 23. Acceptance Criteria

Prototype dianggap memenuhi requirement apabila:

1. Tiga kamera dapat mengambil data inspeksi.
2. CAM-01 dapat melakukan inspeksi Bolt Right.
3. CAM-01 dapat melakukan inspeksi Pin Right.
4. CAM-01 dapat melakukan inspeksi Lekukan Right.
5. CAM-02 dapat melakukan inspeksi Bolt Left.
6. CAM-02 dapat melakukan inspeksi Pin Left.
7. CAM-02 dapat melakukan inspeksi Lekukan Left.
8. CAM-03 dapat melakukan inspeksi Rib.
9. CAM-03 dapat melakukan inspeksi Plate.
10. Sistem menghasilkan OK/NG/INVALID.
11. Hasil inspeksi dapat dikirim melalui REST API.
12. Hasil inspeksi dapat disimpan ke SQL Server.
13. Dashboard dapat menampilkan hasil inspeksi terbaru.
14. Dashboard dapat menampilkan status ketiga kamera.
15. Dashboard dapat menampilkan abnormality.
16. Histori inspeksi dapat ditampilkan.
17. Abnormality image dapat ditelusuri melalui Inspection ID.

---

## 24. Development Roadmap

### Phase 1 — Camera & Data Acquisition

- Finalisasi posisi kamera.
- Konfigurasi RTSP.
- Pengujian FOV.
- Pengujian lighting.
- Pengumpulan image/video.

### Phase 2 — Dataset & Annotation

- Pengumpulan data OK.
- Pengumpulan data NG.
- Annotation bolt.
- Annotation pin.
- Annotation lekukan.
- Annotation rib.
- Annotation plate.
- Data split train/validation/test jika menggunakan supervised learning.

### Phase 3 — Computer Vision Experiment

Eksperimen metode:

- Object Detection.
- Segmentation.
- Geometry.
- Edge/Contour.
- Anomaly Detection.
- metode yang sesuai...

### Phase 4 — Inspection Engine

- Detection.
- Filtering.
- ROI.
- Reference geometry.
- Threshold.
- OK/NG/INVALID logic.
- Abnormality image capture.

### Phase 5 — REST API

- API contract.
- Endpoint.
- Validation.
- Error handling.
- Communication testing.

### Phase 6 — SQL Server

- Database schema.
- Table creation.
- Indexing.
- Data insertion.
- Query optimization.

### Phase 7 — React Dashboard

- Dashboard.
- Camera monitoring.
- KPI.
- Inspection result.
- Abnormality monitoring.
- History.

### Phase 8 — System Integration

```text
Camera
  ↓
AI Vision
  ↓
REST API
  ↓
SQL Server
  ↓
React Dashboard
```

### Phase 9 — Validation

Pengujian menggunakan:

- Normal sample.
- Bolt missing.
- Pin missing.
- Pin displaced.
- Lekukan retak.
- Lekukan ok
- Rib deformation.
- Plate deformation.
- Different lighting.
- Different conveyor positions.
- Camera disturbance.

### Phase 10 — Field Test

- Long-run test.
- False OK analysis.
- False NG analysis.
- Reliability test.
- Performance optimization.
- Final threshold calibration.

---

## 25. Risk Analysis

### 25.1 Camera Angle

Perubahan sudut kamera dapat memengaruhi bentuk objek pada image dan menyebabkan false detection.

### 25.2 Lighting

Perubahan pencahayaan dapat memengaruhi edge, contour, segmentation, dan anomaly detection.

### 25.3 Object Variation

Bolt dan pin dapat memiliki variasi posisi, orientasi, refleksi, dan kondisi permukaan.

### 25.4 Deformation Definition

Definisi deformasi harus dibuat secara objektif.

Contoh:

```text
Deviation <= Threshold → OK
Deviation > Threshold  → NG
```

Threshold harus berasal dari data dan kebutuhan inspeksi, bukan sekadar asumsi.

### 25.5 False OK

False OK merupakan risiko penting karena abnormality dapat lolos dari sistem.

### 25.6 Invalid Inspection

Sistem harus mampu membedakan abnormality aktual dari kondisi ketika kamera/model tidak mampu melakukan inspeksi.

---

## 26. Success Metrics

| KPI | Target |
|---|---|
| Bolt Right Detection Accuracy | TBD |
| Pin Right Detection Accuracy | TBD |
| Lekukan Right Detection Accuracy | TBD |
| Bolt Left Detection Accuracy | TBD |
| Pin Left Detection Accuracy | TBD |
| Lekukan Left Detection Accuracy | TBD |
| Rib Inspection Accuracy | TBD |
| Plate Inspection Accuracy | TBD |
| False OK Rate | TBD |
| False NG Rate | TBD |
| Invalid Rate | TBD |
| Detection Latency | TBD |
| API Response Time | TBD |
| Data Recording Success | TBD |
| Camera Availability | TBD |

Target numerik ditentukan setelah baseline test menggunakan data aktual.

---

## 27. Future Development

Setelah sistem dasar stabil, sistem dapat dikembangkan menjadi:

- Automatic abnormality notification.
- Maintenance notification.
- Abnormality trend analysis.
- Historical dashboard.
- Integration dengan PLC.
- Integration dengan MES.
- Conveyor traceability.
- Maintenance database.
- Automatic inspection report.
- Predictive maintenance.
- Multi-station monitoring.

---

## 28. Final System Concept

Konsep akhir sistem:

```text
                         AI CONVEYOR INSPECTION
                                  |
          +-----------------------+-----------------------+
          |                       |                       |
       CAM-01                  CAM-02                  CAM-03
      RIGHT SIDE              LEFT SIDE              TOP/SURFACE
          |                       |                       |
      +---+---+-----+         +---+---+-----+         +---+---+
      |       |     |         |       |     |         |       |
     Bolt    Pin  lekukan   Bolt    Pin  Lekukan     Rib   Plate
      |       |     |         |       |     |         |       |
      +---+---+-----+         +---+---+-----+         +---+---+
          |                       |                       |
          +-----------------------+-----------------------+
                                  |
                           AI / CV ENGINE
                                  |
                         OK / NG / INVALID
                                  |
                              REST API
                                  |
                           SQL SERVER
                                  |
                           REACT + VITE
                                  |
                         WEB DASHBOARD
```

Sistem pada akhirnya menyediakan satu platform monitoring yang menggabungkan enam inspection point:

**Bolt Right + Pin Right + Lekukan Right + Bolt Left + Pin Left + Lekukan Left + Rib + Plate**

dengan hasil inspeksi yang terdokumentasi, dapat ditelusuri, dan dapat dimonitor melalui web dashboard.
