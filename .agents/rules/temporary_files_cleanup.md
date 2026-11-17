---
trigger: always_on
description: Aturan pengelolaan dan pembersihan otomatis file sekali pakai (scratch, debugging, dan file temporer).
---

## Aturan Pengelolaan File Sekali Pakai (Disposable & Temporary Files)

Untuk menjaga repositori tetap bersih, rapi, dan tidak dipenuhi file sampah sekali pakai (scratch):

1. **Prioritaskan Eksekusi Inline (One-Liner)**:
   - Untuk inspeksi data cepat, pengujian modul, atau diagnostik kecil, jalankan langsung melalui perintah terminal (misalnya: `python -c "..."` atau perintah shell langsung) tanpa perlu membuat file script fisik baru di folder project.

2. **Pembersihan Otomatis & Seketika (Auto Cleanup)**:
   - Jika suatu tugas investigasi benar-benar membutuhkan file script sementara atau file output sementara (seperti `test_*.py`, `temp_*.jpg`, dump file, atau preview image), file tersebut **WAJIB langsung dihapus otomatis** setelah pengujian selesai dijalankan.
   - Jangan pernah meninggalkan sisa file percobaan atau file sementara di direktori project.

3. **Larangan Menyimpan File Scratch**:
   - Dilarang membuat folder `scratch/` baru atau meninggalkan file skrip sekali pakai di dalam workspace.
   - Folder kerja project (`Vision_Engine_Core/`, `Jetson_Vision_Engine/`, root program) hanya diperuntukkan bagi kode sumber produksi, konfigurasi, dan aset resmi konveyor.
