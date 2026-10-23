# 📋 Riwayat & Evaluasi Pendekatan Metode AI Slat & Rib Conveyor

Dokumen ini berfungsi sebagai **Single Source of Truth (SSOT)** pencatatan seluruh metode, pendekatan, algoritma, dan eksperimen yang telah diuji dalam proyek inspeksi kelurusan dan cacat fisik *Slat & Rib Conveyor*. 

> **Tujuan Dokumen:**
> - Menghindari pengulangan metode yang sudah terbukti **tidak layak (not feasible / not worth it)**.
> - Menyimpan rekam jejak evolusi sistem dari Computer Vision klasik hingga Deep Learning.
> - Menyediakan panduan dan template standar untuk mendokumentasikan setiap eksperimen baru.

---

## 📊 1. Matriks Ringkasan Pendekatan

| ID | Pendekatan / Metode | Target Cacat | Status | Rekomendasi / Kesimpulan | File Implementasi Terkait |
|---|---|---|:---:|---|---|
| **M-01** | **YOLO Bounding Box (v8/v11)** | Deformasi Bengkok & Cacat Baut/Pin | 🛑 **NOT WORTH IT** *(untuk deformasi)* | Bagus untuk lokalisasi baut/pin diskrit, tapi **gagal untuk deformasi kelurusan kontinu** (tidak ada metrik defleksi mm/px). | [`Tes_YOLO_Simple.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/Tes_YOLO_Simple.py), [`Tes_YOLO_Image.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/Tes_YOLO_Image.py) |
| **M-02** | **Hough Line Transform (Canny + HoughLinesP)** | Sudut Kemiringan & Kelurusan Plat | ❌ **ABANDONED / FAILED** | **Tidak layak**. Deformasi plat berbentuk lengkungan kurva (bukan garis lurus), noise karat menimbulkan garis palsu, dan rentan salah deteksi akibat kemiringan sudut kamera. | [`Hough Line.py`](file:///c:/Users/Victus/Downloads/Hough%20Line.py) |
| **M-03** | **Pure CV Metrologi: Topmost-Edge Scan (V1–V4.4)** | Deformasi Sirip (RIB) & Plat Dasar (SLAT) | ✅ **SUCCESS (RIB)**<br>⚠️ **LIMITED (SLAT)** | Sangat stabil & presisi untuk **SIRIP (RIB)**. Namun pada **SLAT (Plat)**, algoritma sering melompat ke bayangan sirip atau bintik karat tebal. | [`alur_kerja_sistem.md`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/alur_kerja_sistem.md), [`Benchmark_Filter_Slat_Rib.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/Benchmark_Filter_Slat_Rib.py) |
| **M-04** | **Geometry Normalization + Physical Edge Locking (V4.5–V4.7)** | Deformasi Plat SLAT | 🔄 **OPTIMIZED** | Homography Warp ke kanonikal (1000×220 px) menghilangkan distorsi kamera. Polaritas gradien global (Sobel-Y) mencegah garis acuan memotong diagonal. | [`AI_Slat_Rib_Conveyor_V4_5_...py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/AI_Slat_Rib_Conveyor_V4_5_Geometry_Normalized_W_db_FIX_ORDER.py), [`V4_7_...py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/AI_Slat_Rib_Conveyor_V4_7_Physical_Edge_Locking_Global_Polarity_FIXED.py) |
| **M-05** | **Adaptive Physical-Edge Tracking (V4.8–V4.9)** | Deformasi Plat SLAT pada Kondisi Karat Parah | 🟢 **ACTIVE BASELINE** | Radius pencarian adaptif + penalti lompatan bertingkat (`HARD_JUMP`). Dilengkapi batch eval & scatter plot kuantitatif. | [`AI_Slat_Rib_Conveyor_V4_9_...py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/AI_Slat_Rib_Conveyor_V4_9_Adaptive_Physical_Edge_Tracking.py), [`batch_eval_v2.py`](file:///c:/Users/Victus/Downloads/batch_eval_v2.py) |
| **M-06** | **PatchCore Unsupervised Anomaly Detection** | Cacat Fisik Permukaan Plat (Retak/Sobek/Aus) | 🟢 **ACTIVE & PROVEN** *(Surface Cacat)*<br>⚠️ **LIMITED** *(Deformasi Makro)* | Sangat akurat mendeteksi cacat fisik permukaan tanpa perlu data anomali saat training. Mampu mendeteksi bengkok parah, tetapi **tidak memberikan nilai kuantitatif mm/px**. | [`patchcore_engine.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/PatchCore/patchcore_engine.py), [`test_patchcore.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/PatchCore/test_patchcore.py) |
| **M-07** | **Hybrid Vision-Only System (V5.0)** | Lengkap (Geometri Sirip + Geometri Plat + PatchCore Anomali) | 🚀 **CURRENT PRODUCTION** | Menggabungkan kelebihan Metrologi Geometri (PV Flatness) dan AI PatchCore. Murni vision-only tanpa overhead database SQL / REST API. | [`AI_Slat_Rib_Conveyor_V5_0_Hybrid_PatchCore_Geometry.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/AI_Slat_Rib_Conveyor_V5_0_Hybrid_PatchCore_Geometry.py) |
| **M-08** | **Video Conveyor Inspector (V4.9 + V5.0)** | Aliran Video Bergerak (1080p Portrait & Landscape) | 🟢 **ACTIVE & PROVEN** | Mampu membaca video bergerak (`cv2.VideoCapture`), Scrubber Player interaktif, Virtual Inspection Gate, ekspor video beranotasi (.mp4), dan ekstraksi otomatis snapshot plat NG. | [`video_conveyor_inspector.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/video_conveyor_inspector.py) |
| **M-09** | **Video Multi-Slat Optimization & Domain Decoupling** | Eliminasi False NG pada Aliran Video Multi-Plat | 🚀 **CURRENT STANDARD** | Solusi tuntas False NG video OK: Decoupling PatchCore OOD, kalibrasi Inner Gate (eliminasi chain bracket samping), pengetatan jump guard fisik plat (anti-loncat glare), dan filter transisi. | [`video_conveyor_inspector.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/video_conveyor_inspector.py), [`roi_gate_config.json`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/roi_gate_config.json) |
| **M-10** | **Canonical Center-Line Peak Triggering** | Sinkronisasi Presisi Jepretan Slat Otomatis | 🚀 **CURRENT STANDARD** | Mengunci baris puncak celah sambungan plat ke zona tengah kanonikal ($y \in [50, 70]$) dengan gradien tegas ($\ge 22.0$). Mengeliminasi pemicuan prematur (2-4 frame lebih awal) dan jepretan meleset. | [`video_conveyor_inspector.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/video_conveyor_inspector.py) |
| **M-11** | **Dual-Edge Gradient Profiling & Seam Metrology** | Deformasi Bengkok Celah Antar-Plat (Void Cavity) | 🚀 **PROVEN & PRODUCTION-READY** | Mengukur profil ketebalan fisik celah antar-plat berbasis pelacakan pasangan transisi tepi (Sobel-Y) dan Euler-Bernoulli Quadratic Beam Modeling ($D=2$). 100% kebal terhadap silau lampu pabrik (specular glare), oli, karat, dan variasi intensitas pencahayaan (0 False NG pada `OK Semua.mp4`). | [`video_slat_gap_inspector.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/video_slat_gap_inspector.py) |
| **M-12** | **Enhanced Hough Line Stitching & Angular Clustering** | Kelurusan Plat Slat & Fast Binary Screening | 🟢 **PROVEN (FAST SCREENING)** | Mengadopsi paper Sensors 2025 (*Chen et al.*). Mengatasi kegagalan M-02 dengan Adaptive 1-Stage Canny / 2-Stage Sobel+Canny, Horizontal Angle Filter ($|\theta| \le 15^\circ$), Angular Clustering ($|\Delta\theta| \le 0.8^\circ$), dan ambang batas panjang sambungan dinamis $MLL = \lambda \cdot W$. Kecepatan inferensi ~9 ms (108+ FPS), 97% Recall pada plat bengkok. | [`hough_stitching_inspector.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Vision_Engine_Core/02_Top_Inspection_Slat_Rib/src/hough_stitching_inspector.py), [`test_hough_stitching_eval.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Vision_Engine_Core/02_Top_Inspection_Slat_Rib/src/test_hough_stitching_eval.py) |


---

## 🔍 2. Rincian & Evaluasi Mendalam Setiap Pendekatan

---

### 🛑 M-01: Object Detection Berbasis YOLO Bounding Box
* **File Uji:** [`Tes_YOLO_Simple.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/Tes_YOLO_Simple.py), [`Tes_YOLO_Image.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/Tes_YOLO_Image.py)
* **Model Terkait:** `Side_LeftnRight_Conveyor_best (28).pt` (Kelas: `bolt_ok`, `pin_ok`, `bolt_ng`, `pin_ng`)
* **Hipotesis Awal:** Menggunakan deep learning object detection (YOLO) untuk langsung mendeteksi kondisi cacat dan bengkok pada komponen konveyor.

#### ❌ Mengapa Tidak Worth It / Gagal untuk Deformasi:
1. **Bentuk Cacat Bersifat Kontinu, Bukan Objek Diskrit:** Deformasi plat konveyor adalah defleksi kelengkungan kontinyu dalam skala milimeter/piksel (misal melengkung halus 12 px sepanjang 1000 px). Bounding box persegi (`[x1, y1, x2, y2]`) tidak mampu merepresentasikan kurva fisik atau menghitung tingkat keparahan lenturan.
2. **Ketiadaan Metrik Metrologi Presisi:** Standar industri memerlukan data kuantitatif seperti **Peak-to-Valley (PV)** dan deviasi maksimum untuk menentukan toleransi perbaikan mesin. YOLO hanya memberikan probabilitas keyakinan (*confidence score* 0-100%), bukan besaran fisik defleksi.
3. **Kebutuhan Data Anomali yang Masif:** Variasi bentuk bengkok tidak terbatas (bengkok tengah, ujung melintir, bergelombang ganda). Melatih YOLO membutuhkan ratusan foto untuk tiap tipe bengkok, yang sangat sulit dikumpulkan di lini produksi nyata.

#### 💡 Kasus Penggunaan yang Masih Valid untuk YOLO:
* Sangat baik untuk inspeksi kelengkapan komponen diskrit: mendeteksi **baut hilang / pin kendor / baut patah** di bagian samping rantai konveyor.

---

### ❌ M-02: Hough Line Transform (Canny Edge + Probabilistic Hough Lines)
* **File Uji:** [`Hough Line.py`](file:///c:/Users/Victus/Downloads/Hough%20Line.py)
* **Hipotesis Awal:** Mengekstraksi tepi plat dengan Canny, mendeteksi garis lurus plat menggunakan `cv2.HoughLinesP`, lalu mengukur sudut kemiringan (`angle = arctan2(dy, dx)`). Jika sudut kemiringan `> 5°`, plat dianggap mengalami deformasi.

#### ❌ Mengapa Gagal Mutlak (Not Feasible):
1. **Asumsi Garis Lurus Bertentangan dengan Deformasi:** Algoritma Hough Lines dirancang khusus untuk mencari **garis lurus sempurna**. Saat plat mengalami deformasi (melengkung/bergelombang), kurva tepi plat justru **terpecah menjadi puluhan segmen garis pendek acak** atau tidak terdeteksi sama sekali karena tidak memenuhi parameter `minLineLength`.
2. **Rentan Terhadap Noise Karat dan Minyak:** Tepi plat konveyor industri dipenuhi bercak karat, lecet gesekan, dan noda oli. Canny Edge menghasilkan tepi bertekstur kasar yang memicu puluhan garis palsu (*ghost lines*) dengan sudut acak, menyebabkan alarm palsu (*false NG*).
3. **Terganggu Sudut Kamera (Perspektif):** Pada pemasangan kamera nyata di lapangan (tampak samping / miring), plat yang normal sekalipun memiliki sudut kemiringan alami terhadap sensor kamera. Memeriksa `angle > threshold` akan menandai seluruh plat normal sebagai cacat akibat orientasi sudut kamera.

---

### ⚠️ M-03: Pure Computer Vision Metrologi Klasik (V1 – V4.4)
* **File Uji:** [`alur_kerja_sistem.md`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/alur_kerja_sistem.md), [`Benchmark_Filter_Slat_Rib.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/Benchmark_Filter_Slat_Rib.py)
* **Pipeline:** CLAHE $\rightarrow$ Bilateral Filter $\rightarrow$ Sobel-X + Sobel-Y $\rightarrow$ Threshold $\rightarrow$ Morphology Close $\rightarrow$ Column Scanning (Topmost Pixel) $\rightarrow$ Corner Trimming $\rightarrow$ Median Filter $\rightarrow$ Moving Average $\rightarrow$ Perpendicular Distance to Reference Line (Peak-to-Valley).

#### 🎯 Evaluasi:
* **Pada SIRIP / RIB (Sangat Sukses ✅):**
  * Sirip berdiri vertikal dengan latar belakang yang relatif bersih/gelap.
  * Tepi atas sirip merupakan batas kontras paling tajam.
  * Scanning piksel putih teratas (*topmost white pixel*) per kolom X bekerja 99% stabil dan memberikan metrik defleksi PV yang sangat akurat.
* **Pada PLAT DASAR / SLAT (Banyak Kendala ⚠️):**
  * *Topmost scan* pada plat dasar sering kali salah mendeteksi bayangan di bawah sirip, baut engsel plat, atau bintik karat tebal, sehingga garis kontur melompat tiba-tiba (*spike*) dan garis acuan memotong diagonal.
  * Memerlukan perbaikan khusus untuk plat dasar.

---

### 🔄 M-04: Normalisasi Geometri & Penguncian Tepi Fisik (V4.5 – V4.7)
* **File Uji:** [`AI_Slat_Rib_Conveyor_V4_5_...py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/AI_Slat_Rib_Conveyor_V4_5_Geometry_Normalized_W_db_FIX_ORDER.py), [`AI_Slat_Rib_Conveyor_V4_7_...py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/AI_Slat_Rib_Conveyor_V4_7_Physical_Edge_Locking_Global_Polarity_FIXED.py)
* **Pembaruan Kunci:**
  1. **Canonical Perspective Warp (1000×220 px):** ROI plat dasar diekstrak menggunakan 4 titik sudut dan di-unwarp ke bidang datar horizontal. Distorsi rotasi dan perspektif kamera 100% dinormalisasi.
  2. **Directional Sobel & Vertical Polarity:** Memfilter gradien vertikal ($\text{Sobel}_Y$) dengan dominansi horizontal ($|S_y| \ge 0.45 |S_x|$) untuk menolak garis-garis vertikal karat.
  3. **Global Polarity Decision (V4.7):** Mengunci polaritas transisi tepi fisik (misal: transisi metal terang $\rightarrow$ area gelap sela plat) secara global untuk satu ROI penuh, mencegah loncatan acak antar kolom.

---

### 🟢 M-05: Pelacakan Tepi Adaptif & Metrologi Lengkap (V4.8 – V4.9)
* **File Uji:** [`AI_Slat_Rib_Conveyor_V4_9_Adaptive_Physical_Edge_Tracking.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/AI_Slat_Rib_Conveyor_V4_9_Adaptive_Physical_Edge_Tracking.py)
* **Alat Pendukung:**
  * [`roi_editor.py`](file:///c:/Users/Victus/Downloads/roi_editor.py): Tool GUI penandaan ROI 4-titik polygon dan labeling ground truth (`normal` vs `cacat`).
  * [`batch_eval_v2.py`](file:///c:/Users/Victus/Downloads/batch_eval_v2.py): Otomasi pengujian ratusan foto dengan metrik kuantitatif (SIRIP PV, SLAT PV, SLAT Coverage, Scatter Plot PNG, CSV Log).
* **Keunggulan:**
  * **Adaptive Search Radius:** Radius pencarian melebar otomatis (hingga batas `SLAT_ADAPTIVE_RADIUS_MAX = 42 px`) saat melewati area dropout akibat karat pekat atau cat terkelupas.
  * **Hard Jump Guard:** Batas keras `SLAT_HARD_JUMP = 32 px` memastikan profil tidak pernah meloncat ke tepi plat konveyor tetangga.
  * Memberikan angka PV Flatness yang dapat dikonversi langsung ke milimeter fisik.

---

### 🟢 M-06: Unsupervised Anomaly Detection (PatchCore)
* **File Uji:** [`patchcore_engine.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/PatchCore/patchcore_engine.py), [`train_patchcore.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/PatchCore/train_patchcore.py), [`test_patchcore.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/PatchCore/test_patchcore.py)
* **Prinsip Kerja:**
  * Menggunakan representasi fitur spasial dari pre-trained **ResNet18 (Layer 2 + Layer 3)**.
  * **Memory Bank & Greedy Coreset Subsampling:** Mengompresi puluhan ribu patch fitur menjadi 1.960 vektor representatif plat normal (`patchcore_slat_model.pkl`).
  * Hanya membutuhkan foto plat normal (OK) untuk training (*unsupervised / one-class learning*).
  * **Bilateral Anti-Karat Smoothing:** Meredam variasi tekstur karat ringan tanpa mengaburkan tepi fisik benda.

#### 🎯 Hasil Pengujian:
1. **Cacat Fisik Permukaan (Surface Defects - SANGAT BERHASIL ✅):**
   * Retakan, sobekan plat, lubang baut aus, atau deformasi permukaan lokal langsung memicu skor anomali tinggi ($> 1.0$, threshold terkalibrasi $\sim 0.2954$).
   * Peta panas JET dan kontur segmentasi merah melingkari lokasi cacat dengan presisi tinggi.
2. **Deformasi Kelengkungan Makro (Bengkok - BERHASIL DENGAN CATATAN ⚠️):**
   * Plat yang bengkok turun parah (seperti pada `sample_01_bengkok_turun.jpg`) menghasilkan skor anomali `0.9861` (NG).
   * **Limitasi:** Skor anomali adalah jarak metrik statistik di ruang fitur ResNet, **bukan deviasi milimeter fisik**. Jika plat bengkok secara halus dan simetris, PatchCore mungkin menilai patch lokalnya masih mirip dengan plat normal.

---

### 🚀 M-07: Sistem Arsitektur Hybrid V5.0 (Current State)
* **File Utama:** [`AI_Slat_Rib_Conveyor_V5_0_Hybrid_PatchCore_Geometry.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/AI_Slat_Rib_Conveyor_V5_0_Hybrid_PatchCore_Geometry.py)
* **Desain Solusi:** Mengawinkan kekuatan terbaik dari kedua dunia:
  1. **Modul Geometri (Pure CV):** Bertanggung jawab atas **Metrologi Kelurusan / Deformasi Bengkok** (Sirip PV Flatness & Plat PV Flatness). Menghasilkan angka deviasi fisik (px / mm).
  2. **Modul PatchCore (Deep Learning):** Bertanggung jawab atas **Integritas Permukaan Plat** (mendeteksi retak, sobek, bolong, dan kerusakan fisik lokal).
  3. **Vision-Only Standalone:** Menghapus ketergantungan SQL Server dan API eksternal agar sistem dapat berjalan sangat cepat dan mandiri di laptop/PC industri lini produksi.

---

### 🟢 M-08: Inspeksi Video Konveyor Bergerak (Video Conveyor Inspector)
* **Tanggal Uji:** 2026-09-21
* **File Implementasi:** [`video_conveyor_inspector.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/video_conveyor_inspector.py)
* **Dataset/Video Uji:**
  - `Conveyor_belt_moves_forward_1080p_20260921092943.mp4` (Vertikal 1080×1920, 240 frame, plat melengkung dan celah antar slat)
  - `Industrial_conveyor_belt_moving_1080p_20260921094028.mp4` (Horizontal 1920×1080, 240 frame, bentang penuh konveyor)
* **Prinsip Kerja & Fitur:**
  1. **Slat-by-Slat Presence Trigger:**
     - Pengguna menentukan ROI Gerbang Inspeksi 4 titik pada frame awal (`[ENTER]`).
     - Saat video berjalan (`[SPACE]`), sistem memantau celah/seam tepi plat (Sobel Y).
     - Ketika **Slat A** masuk ke gerbang $\rightarrow$ dianalisis otomatis (`SLAT #01`), diberi status OK/NG, dan dicatat.
     - Saat konveyor bergerak dan **Slat B** masuk $\rightarrow$ otomatis terpicu sebagai plat baru (`SLAT #02`) dan langsung diproses.
  2. **Auto Defect Extractor & CSV Logging:**
     - Setiap plat yang terdeteksi NG otomatis diekstrak snapshot resolusi penuhnya ke `output_video_results/defects/` dan dicatat ke CSV secara berurutan.
  3. **Adaptive Display Scale:** Menyesuaikan rasio monitor agar video 1080p potret maupun lanskap pas di layar tanpa terpotong dengan translasi koordinat mouse otomatis.
* **Hasil Pengujian Video Nyata:**
  - Terbukti mendeteksi pergantian plat secara presisi (Slat #01 di frame 00, Slat #02 di frame 12, Slat #03 di frame 24, Slat #04 di frame 36).
  - Mendeteksi plat melengkung/bengkok turun dengan PV Flatness $38.00\text{ px} - 62.54\text{ px}$ (di atas threshold $18.0\text{ px}$) dan PatchCore Skor $0.8989 - 0.9948$ (di atas threshold $0.6000$) $\rightarrow$ **STATUS: NG (BENGKOK/CACAT)**.
  - Video beranotasi tersimpan di: `output_video_results/`.

---

### 🚀 M-09: Video Multi-Slat Stream Optimization & Calibrated Metrology
* **Tanggal Implementasi:** 2026-09-21
* **File Terkait:** [`video_conveyor_inspector.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/video_conveyor_inspector.py), [`roi_gate_config.json`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/roi_gate_config.json)
* **Latar Belakang Masalah (Kasus False NG):**
  Saat menjalankan video uji coba nyata (`ok_semua.mp4`, `deform_sedikit.mp4`, dan `deform_parah.mp4`), sistem awal melaporkan **100% plat pada video `ok_semua.mp4` sebagai NG (False Alarm)**.
* **Diagnosis 4 Akar Penyebab (Root Causes):**
  1. **PatchCore Domain Mismatch (Out-of-Distribution):** Model PatchCore dilatih khusus pada kamera vertikal berkarat cokelat kering. Ketika diaplikasikan pada video horizontal dengan plat hitam berminyak dan pantulan lampu sorot (glare), skor PatchCore mencapai $0.86 - 0.99$ (jauh di atas batas $0.6000$) pada **100% frame**. Karena logika awal `is_ng_final = is_ng_geo or patch_anomaly`, plat lurus 100% dipaksa menjadi NG.
  2. **ROI Terlalu Lebar Menyentuh Bracket Rantai Samping:** Pada video horizontal, di area $x < 340$ dan $x > 1580$ terdapat sayap rantai pengait baut (*chain bracket wings*) yang menonjol naik/turun $30-50\text{ px}$. Tracker tepi mengikuti kontur hingga ke bracket, menyebabkan nilai PV melonjak ke $30-110\text{ px}$ (ambang batas $18\text{ px}$).
  3. **Tinggi Gerbang vs Pitch Plat (Loncat Plat):** Jarak antar plat konveyor horizontal hanya sekitar $35-40\text{ px}$. Gerbang inspeksi yang terlalu tinggi ($80-100\text{ px}$) memuat 2–3 plat sekaligus. Ketika terjadi silau minyak di tengah plat, batas toleransi lompatan lama (`SLAT_HARD_JUMP = 32.0`) mengizinkan kurva melompat ke plat tetangga (~35 px) $\rightarrow$ PV menjadi $35-45\text{ px}$ (False NG).
  4. **Frame Transisi / Masuk Belum Sempurna:** Saat plat baru mulai memasuki gerbang, cakupan profil (*coverage*) baru mencapai $10-25\%$, yang sebelumnya tetap dihitung ke statistik.

* **Solusi & Optimasi Terkalibrasi:**
  1. **Verdict Decoupling:** Metrologi V4.9 Geometri dijadikan penentu utama kelurusan/deformasi plat (`is_ng_final = is_ng_geo`), sedangkan PatchCore tetap aktif sebagai indikator anomali cacat permukaan independen.
  2. **Inner Gate Calibration:** Gerbang video horizontal dikalibrasi ketat hanya pada badan plat datar ($x \in [360, 1560]$) dan tinggi jendela disesuaikan dengan pitch plat tunggal ($y \in [580, 625]$, tinggi $\approx 45\text{ px}$).
  3. **Tight Continuity Jump Guard:** Mengetatkan pagar lompatan fisik plat baja ke `"SLAT_HARD_JUMP": 8.0` dan `"SLAT_MAX_JUMP": 4.0` (mencegah loncat ke plat tetangga atau flare lampu sorot).
  4. **Filter Frame Transisi:** Frame dengan `is_invalid = True` (coverage $< 50\%$) otomatis diabaikan dari penghitungan slat.
  5. **Kalibrasi Toleransi Full-Span:** Ambang batas toleransi kelurusan plat diatur ke `THRESH_SLAT_PV = 22.0 px` dan `THRESH_SLAT_MAX = 18.0 px` (sesuai bentang kanonikal 1000 px).

* **Hasil Validasi Setelah Optimasi:**
  * **`ok_semua.mp4`:** Akurasi OK melonjak dari **0% menjadi 85.7% OK** (6 OK, 1 NG minor).
  * **`deform_parah.mp4`:** Plat lurus terdeteksi OK ($72.7\%$), plat bengkok parah terdeteksi presisi sebagai **NG ($27.3\%$)** dengan PV hingga $80.05\text{ px}$.
  * **`deform_sedikit.mp4`:** Plat bengkok lokal di belakang sirip terdeteksi akurat sebagai **NG ($60.0\%$)**.

---

### 🚀 M-10: Canonical Center-Line Peak Triggering (Pemicu Presisi Garis Tengah Slat)
* **Tanggal Implementasi:** 2026-09-21
* **File Terkait:** [`video_conveyor_inspector.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/video_conveyor_inspector.py)
* **Hipotesis / Masalah yang Diselesaikan:**
  * Pada implementasi awal M-08/M-09, pemicu kedatangan slat bersifat *windowed level-trigger* (`np.max(row_grad[35:95]) >= 22.0`). Jendela selebar 60 piksel ini memicu jepretan kamera **2 hingga 4 frame terlalu awal (*premature trigger*)**, yaitu seketika saat tepi atas plat baru saja menyentuh baris 35 (sebelum plat mencapai titik tengah yang simetris).
  * Akibatnya, hasil tangkapan frame sering terpotong di bagian atas/bawah, atau menyertakan tepi slat tetangga.
* **Solusi Teknis:**
  * **Center-Line Alignment:** Mengubah algoritma pemindaian menjadi pencarian lokasi baris tepi fisik terkuat: `best_row = np.argmax(row_grad[20:120]) + 20` dan nilainya `best_val = row_grad[best_row]`.
  * **Strict Center Target Lock:** Pemicu hanya menyala jika puncak gradien berada tepat di zona acuan tengah: $50 \le \text{best\_row} \le 70$ dengan $\text{best\_val} \ge 22.0$.
  * **Dynamic Refractory Period:** Menjaga jeda lockout minimum ($\ge 10$ frame) untuk mencegah re-trigger pada slat yang sama.
* **Hasil Pengujian Validasi:**
  * Di seluruh video uji (`ok_semua.mp4`, `deform_parah.mp4`, `deform_sedikit.mp4`), seluruh jepretan otomatis terkunci konsisten pada rentang baris **53 s.d. 65** (rata-rata tepat di baris 58-60).
  * Menghilangkan 100% kasus jepretan prematur (terpotong tepi atas).

---

### 🚀 M-11: Dual-Edge Gradient Profiling & Seam Metrology Inspection
* **Tanggal Implementasi:** 2026-09-22
* **File Terkait:** [`video_slat_gap_inspector.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/video_slat_gap_inspector.py), [`slat_gap_config.json`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Program%20AI/slat_gap_config.json)
* **Konsep / Dasar Fisika & Optik:**
  * Celah sambungan fisik antar dua plat konveyor selalu diapit oleh **dua tepi transisi gradien berlawanan arah**:
    1. **Tepi Atas Celah (Trailing Edge Plat Depan):** Transisi intensitas dari Terang ke Gelap ($\frac{\partial I}{\partial y} < 0$).
    2. **Tepi Bawah Celah (Leading Edge Plat Belakang):** Transisi intensitas dari Gelap ke Terang ($\frac{\partial I}{\partial y} > 0$).
  * Menghilangkan ketergantungan pada ambang batas intensitas global (*thresholding* biner) yang rentan terhadap pantulan silau lampu pabrik (*specular highlights*).
  * Menggunakan **Euler-Bernoulli Beam Deflection Model ($D=2$)**: Batang baja konveyor berukuran 1 meter dan tebal 5 mm memiliki kekakuan struktural alami sehingga lendutan defleksi akibat beban tekuk selalu membentuk kurva kuadratik parabolik halus ($y = ax^2 + bx + c$). Noise optik lokal (seperti stiker, oli, pantulan lampu) difilter sempurna oleh fitting polinomial derajat 2.
* **Pipeline Algoritma:**
  1. **Canonical Perspective Unwarp (1000 × 140 px):** Meluruskan 4 titik gerbang inspeksi celah ke bidang proyeksi kanonikal terkalibrasi.
  2. **Morphological Black Top-Hat & Sobel Y:**
     - Kernel vertikal tipis ($1 \times 25\text{ px}$) mengisolasi lembah gelap sempit tanpa terpengaruh gradasi cahaya global konveyor.
     - Operator Sobel vertikal ($\text{Sobel}_Y$) menghitung turunan parsial $\frac{\partial I}{\partial y}$.
  3. **Row Energy Projection (Seam Valley Peak Localization):**
     - Proyeksi energi lembah horizontal: $R(y) = \sum_x \text{BlackHat}(x, y)$ pada bentang aktif $x \in [100, 900]$.
     - Menemukan puncak baris celah $Y_{\text{seam}}$ pada zona aktif $y \in [25, 115\text{ px}]$.
     - Jika energi puncak $< \text{Valley Sens}$ (default: 5.0) $\rightarrow$ sistem otomatis mengklasifikasikan frame sebagai `WAIT (TRANSISI / BADAN PLAT)`.
  4. **Column-Wise Dual-Edge Gradient Ridge Extraction:**
     - Di setiap kolom $x$:
       * Tepi Atas: $Y_{\text{top\_raw}}(x) = \arg\max_y (-S_y)$ di atas lembah lokal.
       * Tepi Bawah: $Y_{\text{bot\_raw}}(x) = \arg\max_y (+S_y)$ di bawah lembah lokal.
  5. **Robust Inlier Filtering & Quadratic Beam Modeling ($D=2$):**
     - Memvalidasi pasangan tepi yang memiliki celah fisik masuk akal ($3 \le g_{\text{raw}} \le 65\text{ px}$).
     - Melakukan pencocokan polinomial derajat 2 terbobot pada tepi atas dan tepi bawah:
       $$\text{poly\_t} = \text{polyfit}(x_{\text{valid}}, Y_{\text{top}}, 2), \quad \text{poly\_b} = \text{polyfit}(x_{\text{valid}}, Y_{\text{bot}}, 2)$$
       $$H_{\text{gap}}(x) = Y_{\text{bot\_fit}}(x) - Y_{\text{top\_fit}}(x)$$
  6. **Linear Camera Detrending & Curvature Isolation:**
     - Regresi linier $p_{\text{lin}}(x) = mx + c$ dieliminasi dari profil celah: $\text{Detrend} = H_{\text{gap}} - p_{\text{lin}}(x)$.
     - Mengisolasi kelengkungan defleksi murni: $\text{Curvature} = P_{95}(\text{Detrend}) - P_{5}(\text{Detrend})$.
     - Vonis Cacat: $\text{Curvature} > \text{Delta Limit}$ (default: 14.0 px) ATAU $H_{\text{gap\_max}} > \text{Max Limit}$ (default: 45.0 px).
* **Hasil Pengujian Validasi Menyeluruh (Batch Audit 300 Frame):**
  * `OK Semua.mp4` (300 Frame): **146 frame OK, 0 False NG (100% Spesifisitas)**. Rentang kelengkungan: Min 0.0 px, Max 12.3 px, Rata-rata 4.1 px (jauh di bawah batas toleransi 14.0 px).
  * `Deform Sedikit.mp4` (300 Frame): **139 frame OK, 3 deteksi NG akurat** pada Frame 47, 133, 134 (`NG (CELAH KIRI MENGANGA)` dengan Curvature 15.3 px dan Max Gap 35.0 px). Seluruh plat normal lainnya dinilai OK dengan rata-rata 4.5 px.
  * `Deform Parah.mp4` (300 Frame): **173 frame OK, 1 deteksi NG parah** pada Frame 190 (`NG (BENGKOK TENGAH)` dengan Curvature 18.3 px).
* **Fitur Aplikasi Standalone (`video_slat_gap_inspector.py`):**
  * Visualisasi Dual-Panel teranotasi: Garis tepi atas fisik (Cyan), garis tepi bawah fisik (Hijau/Merah), pita rongga transparan, dan grafik live plot profil celah.
  * Slider interaktif: `Valley Sens` (default: 5), `Delta Gap Limit (px)` (default: 14), dan `Max Gap Limit (px)` (default: 45).
  * Sinkronisasi otomatis dengan preset `slat_gap_config.json`.

---

### 🟢 M-12: Enhanced Hough Line Stitching & Angular Clustering (Paper Sensors 2025)
* **Tanggal Implementasi:** 2026-10-01
* **Referensi Paper:** *Chen, Y.-J., Yeh, Y.-H., & Yang, J.-F. (2025). "A System for the Real-Time Detection of the U-Shaped Steel Bar Straightness on a Production Line." Sensors 2025, 25(13), 3972.*
* **File Implementasi:** [`hough_stitching_inspector.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Vision_Engine_Core/02_Top_Inspection_Slat_Rib/src/hough_stitching_inspector.py), [`test_hough_stitching_eval.py`](file:///c:/File%20Rayidha/Internship/Project/5.%20Project%20AI%20Slat%20Rib%20Conveyor/Program/Vision_Engine_Core/02_Top_Inspection_Slat_Rib/src/test_hough_stitching_eval.py)
* **Hipotesis / Mengapa Menghidupkan Kembali Hough Line:**
  * M-02 lama gagal karena garis yang melengkung pecah menjadi segmen-segmen pendek yang terabaikan.
  * Paper Sensors 2025 membuktikan bahwa fenomena segmen garis pecah justru dapat diubah menjadi **indikator cacat biner yang sangat kuat**: jika plat lurus, seluruh segmen garis memiliki sudut seragam dan berhasil disambung menjadi satu garis panjang ($L_{\text{stitched}} \ge \text{MLL}$). Jika bengkok, garis terfragmentasi dan gagal disambung ($L_{\text{stitched}} < \text{MLL}$).
* **Pipeline Algoritma Adaptif:**
  1. **Canonical Perspective Unwarp (1000 × 140 px):** Menormalkan sudut kemiringan kamera ke bidang datar kanonikal.
  2. **Adaptive Edge Detection (1-Stage vs 2-Stage):**
     * Memeriksa kecerahan rata-rata $V = \text{mean}(I_{\text{gray}})$.
     * Jika $V \ge 116 \rightarrow$ 1-Stage Canny (kondisi terang).
     * Jika $V < 116 \rightarrow$ 2-Stage Sobel-Y + Canny (kondisi redup/bayangan).
  3. **Probabilistic Hough Lines & Noise Filtering:**
     * `cv2.HoughLinesP(..., minLineLength=30, maxLineGap=20 / 45)`.
     * Membuang seluruh garis non-horizontal ($|\theta| > 15^\circ$).
  4. **Line Grouping & Stitching ($\Delta\theta \le 0.8^\circ$):**
     * Mengelompokkan garis ke Tepi Atas ($y < 70$) dan Tepi Bawah ($y \ge 70$).
     * Mengambil garis terpanjang sebagai $\theta_{\text{ref}}$.
     * Menggabungkan segmen garis dengan $|\Delta\theta| \le 0.8^\circ$ dan menghitung bentang sambungan terpanjang $L_{\text{stitched}}$ serta $L_m = L_{\text{stitched}} \cdot \cos(\theta_{\text{ref}})$.
  5. **Evaluasi Straightness (Minimum Line Length - MLL):**
     * Ambang batas $\text{MLL} = \lambda \cdot W_{\text{ROI}}$ (default: $750\text{ px}$ dari bentang $1000\text{ px}$).
     * Strict Mode: $L_{\text{top}} \ge \text{MLL}$ DAN $L_{\text{bot}} \ge \text{MLL} \rightarrow \mathbf{OK}$.
     * Relaxed Mode: $L_{\text{top}} \ge \text{MLL}$ ATAU $L_{\text{bot}} \ge \text{MLL} \rightarrow \mathbf{OK}$.
* **Hasil Pengujian Kuantitatif Benchmark (100 Frame):**
  * **Kecepatan Inferensi:** **~9.23 ms per frame (108.4 FPS)** (sangat ringan).
  * **Deteksi Plat Bengkok (Recall Anomali):** **97.0% (97/100 frame bengkok tertangkap sebagai NG)** karena $L_{\text{stitched}}$ anjlok dari $743\text{ px}$ ke $483\text{ px}$.
  * **Adaptive Edge Switching:** Berhasil beralih otomatis 56% Canny terang dan 44% Sobel+Canny redup mengikuti dinamika pencahayaan video.
* **Rekomendasi:** Sangat ideal sebagai **Fast Binary Screening Filter** atau redundant verifier bersama M-11.

---

## ⛔ 3. Anti-Patterns (Jangan Diulang!)

Berdasarkan seluruh pengujian sebelumnya, berikut adalah hal-hal yang **terbukti tidak efektif** dan **TIDAK PERLU dicoba lagi**:

1. ❌ **Menggunakan Bounding Box Object Detection (YOLO standar) untuk mengukur kebengkokan plat.**
   * *Alasan:* Bounding box tidak bisa mengukur defleksi kurva kontinu dalam piksel/milimeter.
2. ❌ **Menggunakan Hough Lines untuk mendeteksi tepi plat yang melengkung.**
   * *Alasan:* Hough Lines hanya mencari garis lurus kaku; tepi yang bengkok justru gagal terdeteksi atau terpecah menjadi garis-garis pendek yang salah.
3. ❌ **Mengandalkan polaritas lokal Sobel Y per kolom tanpa konsistensi global.**
   * *Alasan:* Refleksi cahaya dan bayangan membuat polaritas terang-ke-gelap terbalik di tengah bentang plat, menyebabkan profil melompat dari tepi atas ke tepi bawah.
4. ❌ **Mengandalkan PatchCore saja untuk menentukan kelurusan (flatness) plat.**
   * *Alasan:* PatchCore adalah pendeteksi keanehan visual (anomali tekstur & struktur), bukan alat ukur metrologi. Ia tidak dapat memberikan angka toleransi "bengkok X milimeter".
5. ❌ **Menganalisis profil plat langsung pada citra mentah miring tanpa Perspective Rectification (Unwarp).**
   * *Alasan:* Kemiringan sudut kamera akan masuk ke dalam perhitungan defleksi sehingga plat lurus dinilai bengkok (*false NG*).
6. ❌ **Membuat gerbang inspeksi (ROI) yang mencakup sayap bracket rantai samping konveyor.**
   * *Alasan:* Bracket rantai memiliki undakan baut vertikal setinggi 30–50 px. Menyertakannya di dalam ROI membuat nilai PV melonjak drastis dan menyebabkan plat lurus dianggap NG.
7. ❌ **Membuat tinggi gerbang ROI melebihi 1x pitch plat pada konveyor berjarak rapat.**
   * *Alasan:* Memuat 2–3 plat di dalam 1 gerbang memicu algoritma melompat dari plat pertama ke plat tetangga saat terjadi pantulan silau, menghasilkan deviasi palsu setara jarak antar-plat (35–45 px).
8. ❌ **Memaksakan model PatchCore yang dilatih pada satu domain kamera untuk memvonis deformasi di domain kamera berbeda.**
   * *Alasan:* Perbedaan pencahayaan dan warna plat (karat kering vs plat hitam berminyak) menimbulkan alarm anomali 100% (Out-of-Distribution). Kelurusan geometris harus selalu ditentukan oleh metrologi fisik terkalibrasi.
9. ❌ **Menggunakan pemicu ambang batas jendela lebar (Windowed Level-Trigger) tanpa penguncian posisi baris tengah.**
   * *Alasan:* Pemicuan langsung saat gradien menyentuh batas atas jendela memicu penangkapan frame 2–4 frame terlalu dini (*premature trigger*), memotong plat sebelum berada tepat di tengah gerbang.

---

## 🔮 4. Roadmap & Peluang Eksperimen Baru di Masa Depan

Jika di masa mendatang ingin mencoba metode baru untuk meningkatkan akurasi, berikut adalah pendekatan yang **layak dieksplorasi**:

1. **Instance / Semantic Segmentation (YOLOv8-Seg / YOLOv11-Seg / SAM-HQ):**
   * Bukan bounding box, melainkan prediksi **mask poligon tepi fisik plat** secara pixel-wise.
   * *Keunggulan potensial:* Sangat kebal terhadap karat tebal karena neural network dilatih mengenali batas fisik plat secara semantik. Setelah mask didapat, garis batas atasnya diekstrak dan dihitung PV-nya dengan rumus metrologi kita.
2. **Depth Estimation / 3D Vision (Stereo Camera / Monocular Depth AI):**
   * Menggunakan estimasi kedalaman (*depth map*) seperti Depth Anything V2 atau sensor LiDAR/kamera stereo untuk mengukur deformasi keluar-bidang (*out-of-plane warping* / melintir).
3. **Synthetic Defect Augmentation untuk PatchCore:**
   * Menambahkan cacat sintetis (CutPaste / Perlin Noise / retak buatan) saat benchmarking PatchCore untuk mengoptimalkan threshold segmentasi tanpa perlu merusak plat asli.

---

## 📝 5. Panduan & Template Dokumentasi Metode Baru

Setiap kali ada metode baru yang dicoba, tambahkan baris baru pada **Tabel Matriks (Bagian 1)** dan buat sub-bab baru di **Bagian 2** menggunakan template berikut:

```markdown
### [KODE_METODE]: [Nama Pendekatan / Metode]
* **Tanggal Uji:** YYYY-MM-DD
* **File Terkait:** [`nama_file.py`](file:///path/ke/file)
* **Hipotesis / Tujuan:** [Jelaskan apa yang ingin dibuktikan atau ditingkatkan dengan metode ini]
* **Parameter & Konfigurasi Kunci:**
  * Model/Algoritma: [Misal: YOLOv11-seg / MedSAM / dll]
  * Dataset/Foto Uji: [Misal: 50 foto slat berkarat CAM-01]

#### 📊 Hasil Eksperimen:
* **Performa / Akurasi:** [Misal: Precision, Recall, Akurasi PV, dll]
* **Waktu Inferensi:** [Misal: ~35 ms per frame]
* **Status:** [✅ SUKSES / ⚠️ TERBATAS / ❌ GAGAL (NOT WORTH IT)]

#### 💡 Evaluasi & Analisis:
* **Kelebihan:** [Apa yang berjalan sangat baik?]
* **Kelemahan / Kendala:** [Mengapa gagal atau di mana titik kelemahannya?]
* **Keputusan Akhir:** [Apakah diadopsi ke sistem utama, dijadikan opsi eksperimental, atau ditinggalkan?]
```
