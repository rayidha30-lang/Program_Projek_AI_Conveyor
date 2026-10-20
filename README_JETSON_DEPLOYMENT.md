# 🚀 Panduan Deployment: NVIDIA Jetson Orin Nano (Ubuntu Linux)
### Sistem Inspeksi AI Slat Rib Conveyor

Dokumen ini berisi panduan lengkap instalasi dependensi, konfigurasi driver hardware, dan langkah menjalankan sistem inspeksi AI pada mini PC edge **NVIDIA Jetson Orin Nano** (arsitektur `aarch64` / ARM64).

---

## 📋 1. Ringkasan Kebutuhan Lingkungan

| Komponen | Spesifikasi / Kebutuhan |
| :--- | :--- |
| **Hardware** | NVIDIA Jetson Orin Nano (Developer Kit / Commercial Carrier Board) |
| **OS** | Ubuntu 20.04 LTS (JetPack 5.1.x) atau Ubuntu 22.04 LTS (JetPack 6.x) |
| **Python** | Python 3.8 (JetPack 5) atau Python 3.10 (JetPack 6) |
| **Akselerator** | NVIDIA Ampere Architecture GPU (CUDA 11.4 / 12.x, TensorRT 8.x) |
| **Kamera** | USB Industrial Camera (V4L2) atau MIPI CSI Camera (GStreamer NVMM) |

---

## ⚡ 2. Cara Cepat: Setup Otomatis (Rekomendasi)

Kami telah menyediakan skrip automasi `setup_jetson.sh` yang otomatis mendeteksi versi Ubuntu/JetPack, mengunduh wheel PyTorch resmi NVIDIA, dan menginstal dependensi sistem:

```bash
# 1. Masuk ke direktori Program pada Jetson
cd /path/ke/folder/Program

# 2. Beri izin eksekusi pada script setup
chmod +x setup_jetson.sh

# 3. Jalankan script setup
./setup_jetson.sh
```

---

## 🛠️ 3. Panduan Instalasi Manual Langkah-demi-Langkah

Jika Jetson Anda berada dalam jaringan tertutup pabrik (tanpa akses langsung atau memerlukan proxy), ikuti langkah berikut:

### Langkah 1: Instal Paket Sistem Ubuntu (APT)
Di Jetson, beberapa modul Python membutuhkan header library C++ dan driver sistem:
```bash
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-tk \
    build-essential \
    libopenblas-dev \
    libopenmpi-dev \
    libjpeg-dev \
    zlib1g-dev \
    unixodbc \
    unixodbc-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    v4l-utils \
    curl \
    git
```

> **Catatan Penting:** 
> - `python3-tk` wajib agar fitur file dialog explorer GUI tidak error saat menekan tombol `[O]` / `[F]`.
> - `unixodbc` & `unixodbc-dev` wajib agar modul `pyodbc` dapat terkompilasi dan menghubungkan backend ke database SQL Server.

---

### Langkah 2: Instalasi PyTorch Khusus NVIDIA Jetson (CUDA)
> ⚠️ **PERINGATAN KRUSIAL:**
> Jangan pernah menjalankan `pip install torch torchvision` langsung dari PyPI di Jetson! PyPI standar tidak menyediakan wheel GPU CUDA untuk arsitektur ARM64 Jetson. Gunakan wheel resmi NVIDIA:

#### Opsi A: Untuk JetPack 5.1.x (Ubuntu 20.04 - Python 3.8)
```bash
pip3 install --no-cache-dir https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
pip3 install --no-cache-dir --no-deps torchvision==0.16.1
```

#### Opsi B: Untuk JetPack 6.0 / 6.1 (Ubuntu 22.04 - Python 3.10)
```bash
pip3 install --no-cache-dir https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0+nv24.05-cp310-cp310-linux_aarch64.whl
pip3 install --no-cache-dir --no-deps torchvision==0.18.0
```

Verifikasi GPU CUDA aktif:
```bash
python3 -c "import torch; print('CUDA Ready:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
# Output diharapkan: CUDA Ready: True | GPU: Orin
```

---

### Langkah 3: Kunci Versi NumPy & Instal requirements.txt
PyTorch dan OpenCV di Jetson ARM64 membutuhkan `numpy < 2.0.0` agar tidak terjadi konflik ABI:
```bash
# 1. Kunci NumPy
pip3 install "numpy>=1.23.5,<2.0.0"

# 2. Instal seluruh modul proyek
pip3 install -r requirements.txt
```

---

### Langkah 4: Setup Driver ODBC SQL Server (Untuk Program AI / db.py)
Jika Jetson perlu mengirim data hasil inspeksi ke Microsoft SQL Server:
```bash
# Tambahkan repositori Microsoft untuk Ubuntu ARM64
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo tee /etc/apt/trusted.gpg.d/microsoft.asc
curl -sSL https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/prod.list | sudo tee /etc/apt/sources.list.d/mssql-release.list

sudo apt-get update
sudo ACCEPT_EULA=Y apt-get install -y msodbcsql18
```

---

## 🏃 4. Cara Menjalankan Modul di Jetson Orin Nano

### A. Menjalankan Inspeksi Tampak Atas Plat (Hough Line Stitching)
Untuk kamera USB / Webcam bawaan:
```bash
python3 Vision_Engine_Core/02_Top_Inspection_Slat_Rib/src/hough_stitching_inspector.py --cam 0
```
Untuk menguji dengan file rekaman video:
```bash
python3 Vision_Engine_Core/02_Top_Inspection_Slat_Rib/src/hough_stitching_inspector.py
```

### B. Menjalankan Inspeksi Sirip Hibrida (Hybrid Rib Inspector)
```bash
python3 Vision_Engine_Core/02_Top_Inspection_Slat_Rib/src/hybrid_rib_inspector.py
```

### C. Menjalankan Inspeksi Baut & Pin Samping (Side Inspector)
```bash
python3 Vision_Engine_Core/01_Side_Inspection_Bolt_Pin_Lekukan/src/side_inspector_pipeline.py
```

### D. Menjalankan Backend REST API Server (Port 5000)
```bash
cd "Program AI"
python3 api.py
```

### E. Menjalankan Web Dashboard Frontend (React Vite)
```bash
cd FE
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```
> Menambahkan argumen `--host 0.0.0.0` memungkinkan Dashboard diakses dari browser PC/laptop lain di jaringan lokal pabrik via `http://<IP_JETSON>:5173`.

---

## 💡 5. Tips Maksimalisasi Performa di Jetson Orin Nano

1. **Aktifkan Power Mode Maksimum & Kunci Clock GPU:**
   ```bash
   sudo nvpmodel -m 0        # Mode daya 15W / 25W MAXN
   sudo jetson_clocks        # Kunci CPU/GPU clock pada frekuensi puncak
   ```

2. **Ekspor Model YOLO ke Format TensorRT Engine (FP16):**
   Model YOLO (`best_v2.pt` atau `best_rib.pt`) akan berjalan **3x hingga 5x lebih cepat** (~5–10 ms/frame) jika diekspor ke TensorRT:
   ```bash
   yolo export model=Vision_Engine_Core/01_Side_Inspection_Bolt_Pin_Lekukan/models/best_v2.pt format=engine half=True device=0
   ```
   Setelah itu, model `.engine` dapat langsung dipanggil di dalam detector:
   `self.model = YOLO("models/best_v2.engine")`

3. **GStreamer Pipeline untuk Kamera MIPI CSI:**
   Jika menggunakan kamera sensor MIPI CSI industri (seperti IMX477 / IMX219), gunakan format pipeline GStreamer hardware acceleration:
   ```python
   pipeline = (
       "nvarguscamerasrc sensor-id=0 ! "
       "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
       "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
   )
   cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
   ```
