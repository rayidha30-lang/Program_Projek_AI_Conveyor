# Synthetic ArUco Video Alignment Simulator V2
## Modul 04_Camera_Auto_Calibration (Reference-Based Offline Video Alignment Engine)

Simulator ini adalah alat kalibrasi dan penyelarasan perspektif kamera offline industri untuk mengoreksi perubahan sudut kamera (misal akibat tersenggol mekanik atau getaran mesin conveyor) kembali ke acuan emas (**Reference View**) menggunakan **Fiducial ArUco Marker** aktual yang tertanam di dalam rekaman video.

---

## 1. Arsitektur Tiga Panel (Tri-Panel Studio)

```text
 ┌───────────────────────┬───────────────────────┬───────────────────────┐
 │   [VIEW 1 - KIRI]     │   [VIEW 2 - TENGAH]   │   [VIEW 3 - KANAN]    │
 │   REFERENCE VIEW      │   CURRENT SHIFTED     │   ALIGNED RESULT      │
 │  (reference_aruco)    │   (shifted_aruco)     │   (Homography Warp)   │
 └───────────────────────┴───────────────────────┴───────────────────────┘
 └───────────────────────────────────────────────────────────────────────┘
  [TELEMETRY BAR]: Status | Inlier Ratio | Reprojection Error | Recording
```

- **[VIEW 1: REFERENCE VIEW]**: Menampilkan video acuan emas (`reference_aruco.mp4`). Memindai dan mengunci koordinat fisik ke-4 marker (ID 0, 1, 2, 3) sebagai baseline paten.
- **[VIEW 2: CURRENT SHIFTED VIEW]**: Menampilkan video dengan sudut kamera yang bergeser (`shifted_aruco.mp4`). Mendeteksi marker aktual frame-by-frame dan mencocokkannya berdasarkan ID.
- **[VIEW 3: ALIGNED RESULT]**: Menampilkan hasil transformasi perspektif (`cv2.warpPerspective`) dari Current Coordinates ke Reference Coordinates.
- **[GHOSTING OVERLAY]**: Memadukan 50% Reference + 50% Aligned untuk memverifikasi keselarasan rel dan struktur mesin secara transparan tanpa bayangan ganda (*no double-vision*).

---

## 2. Fitur Utama Versi 2 (V2)

1. **Smart Dual-Mode (Deteksi Murni vs Simulasi ArUco Virtual)**:
   - **Mode Otomatis (AUTO)**: Jika video yang dimasukkan sudah memiliki ArUco (misal dari CapCut atau stiker fisik di mesin), OpenCV akan mendeteksi marker langsung dari piksel video alami tanpa injeksi apa pun.
   - **Mode Virtual untuk Video Polos**: Jika memuat rekaman konveyor polos pabrik tanpa marker (seperti *Opsi Posisi 2*, *Kamera Posisi 4*, dll.), simulator secara otomatis memproyeksikan 4 ArUco virtual pada rangka rel konveyor secara proyektif murni, sehingga pengguna dapat menguji kalibrasi dan pergeseran kamera pada **semua video apa saja** secara instan.
2. **Multi-Corner Registration (Hingga 16 Pasangan Titik)**:
   - Mengambil 4 sudut (*corners*) dari masing-masing 4 marker (total 16 titik korespondensi) untuk kestabilan estimasi matriks Homography $H$.
3. **Validasi Homography 7-Tahap**:
   - Memeriksa jumlah titik ($\ge 12$), singularitas matriks ($|\det(H)| > 10^{-6}$), rasio inlier RANSAC ($\ge 70\%$), rata-rata *reprojection error* ($\le 4.5\text{ px}$), konveksitas poligon batas frame, dan rasio penskalaan area ($0.40 - 2.50\times$).
   - Menolak matriks ekstrem dan menampilkan status `ALIGNMENT ERROR`.
4. **Perlindungan Terhadap Marker Hilang (Cached Fallback)**:
   - Jika marker tertutup sementara, sistem menggunakan matriks valid terakhir (*Last Known Good Matrix*) dan memberi label eksplisit `[USING CACHED ESTIMATE]`.
5. **Pemilihan Video Dinamis (Native File Explorer & Preset Switcher)**:
   - Tombol `[V]` dan `[B]` membuka dialog Windows Explorer untuk memilih video apa saja langsung saat aplikasi berjalan.
   - Tombol `[N]` beralih instan ke rekaman video konveyor berdurasi panjang (~90 s/d 135 detik).
6. **Ekspor Video & Laporan Otomatis**:
   - `aligned_output.mp4`: Video hasil pelurusan penuh (1080p).
   - `comparison_output.mp4`: Video studio 3-panel lengkap (1920x450).
   - `ghosting_output.mp4`: Video overlay transparan 50/50.
   - `calibration_report.json`: Laporan metrologi berisi koordinat titik acuan, inlier ratio, reprojection error, matriks $H$, dan statistik sesi.

---

## 3. Cara Menjalankan

### A. Eksekusi Standar (Membaca file di folder `data/`):
```powershell
python "Vision_Engine_Core\04_Camera_Auto_Calibration\src\aruco_calibration_simulator.py"
```

### B. Menentukan File Video Spesifik via CLI:
```powershell
python "Vision_Engine_Core\04_Camera_Auto_Calibration\src\aruco_calibration_simulator.py" --ref "data/reference_aruco.mp4" --shifted "C:/Users/Victus/Pictures/Video & Foto Test Slat Rib/Opsi Posisi 2 - Slat Rib.mp4" --record
```

### Opsi Parameter CLI Tambahan:
- `--ref <path>`: Jalur berkas video referensi acuan.
- `--shifted <path>`: Jalur berkas video sudut kamera bergeser.
- `--output-dir <path>`: Direktori tujuan ekspor berkas rekaman dan laporan.
- `--record`: Mulai merekam ketiga berkas video secara otomatis saat aplikasi dibuka.
- `--no-loop`: Berhenti saat video mencapai akhir durasi (tanpa perulangan otomatis).
- `--reproj-thresh <float>`: Batas maksimal *reprojection error* rata-rata (default: `4.5` px).
- `--inlier-thresh <float>`: Batas minimal rasio inlier RANSAC (default: `0.70`).

---

## 4. Kontrol Interaktif (Hotkeys)

Saat jendela simulator aktif, gunakan tombol keyboard berikut:

| Tombol Keyboard | Aksi | Deskripsi |
| :---: | :--- | :--- |
| **`[SPASI]`** | **Play / Pause** | Menjeda atau melanjutkan pemutaran kedua video secara sinkron. |
| **`[V]`** | **Pilih Video View 2** | Membuka jendela Windows File Explorer untuk memilih video apa saja untuk View 2. |
| **`[B]`** | **Pilih Video View 1** | Membuka jendela Windows File Explorer untuk memilih video acuan referensi baru. |
| **`[N]`** | **Ganti Video Cepat** | Beralih instan ke rekaman video konveyor pabrik berdurasi panjang (~135 detik). |
| **`[T]`** | **Toggle Mode ArUco** | Beralih mode: `AUTO` (otomatis sesuai isi video) $\rightarrow$ `ON` (paksa virtual) $\rightarrow$ `OFF` (murni). |
| **`[M]`** | **Ganti Sudut Simulasi** | Mengganti profil sudut pergeseran kamera virtual (Posisi 4, Miring, Zoom, Ekstrem). |
| **`[C]`** | **Goyangan / Getaran Mesin** | Mengaktifkan/menonaktifkan getaran dinamis konveyor pada simulasi. |
| **`[G]`** | **Ghosting Overlay** | Mengaktifkan/menonaktifkan transparansi 50% Reference + 50% Aligned pada View 3. |
| **`[S]`** | **Screenshot** | Menyimpan tangkapan layar studio dan frame hasil rektifikasi ke folder `output/`. |
| **`[E]`** | **Rekam / Export** | Memulai atau menghentikan perekaman ketiga video `.mp4` secara interaktif. |
| **`[F]`** | **Freeze Frame** | Membekukan frame referensi (mengunci baseline acuan). |
| **`[R]`** | **Reset Playback** | Mengulang pemutaran video dari frame 0. |
| **`[Q]` / `[ESC]`** | **Keluar** | Menutup simulator dan secara otomatis menerbitkan `calibration_report.json`. |

---

## 5. Batasan Geometris & Catatan Industri

1. **Asumsi Koplanar (Planar Assumption)**:
   - Planar Homography mengasumsikan keempat marker ArUco menempel pada bidang planar yang sama (atau mendekati satu bidang datar pada rangka mesin konveyor).
2. **Efek Parallax 3D**:
   - Objek yang memiliki perbedaan kedalaman fisik signifikan terhadap bidang marker (misalnya sirip rib atau slat yang menonjol tinggi) dapat mengalami deviasi residu lokal kecil akibat proyeksi perspektif 3D.
3. **Penyelarasan Non-Destruktif**:
   - Area di luar batas sudut pandang kamera asal diisi dengan padding hitam (*black border*) tanpa melakukan cropping paksa yang dapat mengaburkan distorsi geometris.
