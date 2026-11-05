# Modul Kalibrasi Kamera Otomatis (04_Camera_Auto_Calibration)

Modul ini adalah alat pengujian dan simulasi mandiri (*standalone testing wizard*) untuk membuktikan bahwa metode **ArUco Fiducial Marker + Homografi OpenCV `calib3d`** mampu:
1. **Mengunci Geometri Kamera (Kebal Guncangan / Perubahan Sudut / Naik-Turun)**: Mengembalikan citra miring ke tampak atas tegak lurus ($90^\circ$ ortogonal) secara otomatis tanpa perlu campur tangan manusia.
2. **Menstabilkan Pencahayaan (Photometric Gain Normalizer)**: Menyamakan intensitas kecerahan dan kontras menggunakan acuan warna putih marker ArUco.

---

## Cara Menjalankan

Jalankan perintah berikut di terminal:

```bash
python Vision_Engine_Core/04_Camera_Auto_Calibration/src/aruco_calibration_simulator.py
```

Atau tentukan file video lain secara spesifik:
```bash
python Vision_Engine_Core/04_Camera_Auto_Calibration/src/aruco_calibration_simulator.py --video "path/to/video.mp4"
```

---

## Kontrol Interaktif (Hotkeys)

Saat jendela simulator aktif, gunakan tombol berikut:

| Tombol Keyboard | Aksi Simulasi | Keterangan |
| :---: | :--- | :--- |
| **`[W]`** | **Kamera Naik** | Menggeser kamera ke atas ($+Y$ dan sedikit *zoom out*). |
| **`[S]`** | **Kamera Turun** | Menggeser kamera ke bawah ($-Y$ dan sedikit *zoom in*). |
| **`[A]` / `[D]`** | **Kemiringan Tilt** | Memiringkan sudut pandang kamera ke kiri/kanan. |
| **`[O]`** | **Auto-Oscillation (Goyangan Otomatis)** | Mengaktifkan simulasi getaran mesin pabrik secara dinamis ($\pm 40\text{ px}$). |
| **`[L]`** | **Siklus Pencahayaan** | Mensimulasikan lampu redup ($0.6\times$) $\leftrightarrow$ silau ($1.4\times$) $\leftrightarrow$ normal. |
| **`[X]`** | **Uji Oklusi / Noda Oli** | Menutup Marker #2 dengan noda hitam untuk menguji proteksi *Last Known Good Matrix*. |
| **`[R]`** | **Reset Perturbasi** | Mengembalikan posisi kamera dan lampu ke kondisi normal. |
| **`[SPACE]`** | **Play / Pause** | Menjeda atau melanjutkan pemutaran video. |
| **`[Q]` / `[ESC]`** | **Keluar** | Menutup aplikasi. |
