# Panduan Review Dataset Rib di PC

Script ini memberi judgment manual pada sampel yang memiliki dua bukti gambar: raw dan AI-overlay. Script tidak mengubah threshold atau keputusan detektor.

## Menjalankan

Jalankan dari folder proyek:

```powershell
python scripts/review_rib_dataset.py --db Database/AI_Slat_Rib_Conveyor.db --reviewer NamaAnda
```

Sampel ditampilkan dari yang tertua dan belum direview. Jika belum ada sampel dengan dua gambar bukti, script akan selesai tanpa menulis label apa pun.

## Tombol dan judgment

- `O`: **OK** — komponen fisik terlihat normal.
- `N`: **NG** — defek yang ditampilkan benar-benar terlihat pada bukti.
- `I`: **INVALID** — kualitas bukti tidak memungkinkan judgment, misalnya blur, objek tertutup, atau framing salah.
- `S`: skip sementara tanpa menulis review.
- `Q` atau `Esc`: keluar.

Masukkan catatan singkat setelah memilih `O`, `N`, atau `I` bila perlu. Jangan memberi label pada sampel yang gambarnya tidak terbuka; perbaiki bukti atau lewati sampelnya.
