# Graph Report - Program  (2026-09-13)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 232 nodes · 410 edges · 14 communities
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 13 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Program AI/AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py
- AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py
- AI_Slat_Rib_Conveyor_V4_3_Robust_Profile.py
- Tes_YOLO_Image.py
- AI_Slat_Rib_Conveyor_V4_2_Robust.py
- Tes_Garis_Acuan_PureCV_V4_w_Rib&Slat.py
- Benchmark_Filter_Slat_Rib.py
- Tes_Garis_Acuan_PureCV_V5_w_Rib&Slat_Filter_Benchmark.py
- Tes_Garis_Acuan_PureCV_V7_w_Rib&Slat_Filter_Benchmark.py
- Tes_Garis_Acuan_PureCV_V6_w_Rib&Slat_Filter_Benchmark.py
- Menggambar garis acuan, kontur deteksi, dan vektor lendutan terbesar.
- Tes_Garis_Acuan_PureCV_V1.py
- Tes_Garis_Acuan_PureCV_V2.py

## God Nodes (most connected - your core abstractions)
1. `interactive_multi_roi_session()` - 14 edges
2. `interactive_multi_roi_session()` - 14 edges
3. `interactive_multi_roi_session()` - 14 edges
4. `interactive_multi_roi_session()` - 11 edges
5. `interactive_multi_roi_session()` - 10 edges
6. `interactive_multi_roi_session()` - 10 edges
7. `interactive_multi_roi_session()` - 10 edges
8. `interactive_multi_roi_session()` - 10 edges
9. `extract_slat_profile_robust_v44()` - 9 edges
10. `extract_slat_profile_robust()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `Robust SLAT Physical Profile Extraction V4.3. Pipeline: ROI -> CLAHE ->…` --rationale_for--> `extract_slat_profile_robust()`  [EXTRACTED]
  AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py → Program AI/AI_Slat_Rib_Conveyor_V4_3_Robust_Profile.py
- `Metrologi SIRIP: P-V + Max Abs Deviation (dual threshold).` --rationale_for--> `analyze_sirip_pv()`  [EXTRACTED]
  AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py → Program AI/AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py
- `Profil SIRIP: topmost white pixel dari Morph Close. Algoritma asli…` --rationale_for--> `extract_clean_contour()`  [EXTRACTED]
  AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py → Program AI/AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py
- `Outlier rejection KONSERVATIF: 1. Isolated spike removal (jauh dari KEDUA…` --rationale_for--> `_reject_slat_outliers()`  [EXTRACTED]
  AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py → Program AI/AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py
- `Temukan local maxima gradient di range [y_min, y_max]. Return list (y,…` --rationale_for--> `_find_col_candidates()`  [EXTRACTED]
  AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py → Program AI/AI_Slat_Rib_Conveyor_V4_3_Robust_Profile.py

## Import Cycles
- None detected.

## Communities (14 total, 0 thin omitted)

### Community 0 - "Program AI/AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py"
Cohesion: 0.10
Nodes (32): analyze_sirip_pv(), analyze_slat_profile(), choose_file_dialog(), clean_path_string(), _ctx_bounds(), draw_sirip_visuals(), draw_slat_visuals(), extract_clean_contour() (+24 more)

### Community 1 - "AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py"
Cohesion: 0.12
Nodes (26): analyze_sirip_pv(), analyze_slat_profile(), choose_file_dialog(), clean_path_string(), _ctx_bounds(), draw_sirip_visuals(), draw_slat_visuals(), extract_clean_contour() (+18 more)

### Community 2 - "AI_Slat_Rib_Conveyor_V4_3_Robust_Profile.py"
Cohesion: 0.12
Nodes (26): Profil SIRIP: topmost white pixel dari Morph Close. Algoritma asli…, Metrologi SIRIP: P-V + Max Abs Deviation (dual threshold)., Score = 55% edge_strength + 45% position_continuity. Penalti berat jika…, Metrologi SLAT V4.3: P-V + Max Abs Deviation. Reference line = endpoint-to-…, Visualisasi SLAT V4.3: ROI outline – cyan tipis Reference line – ORANGE Profile…, analyze_sirip_pv(), analyze_slat_profile(), choose_file_dialog() (+18 more)

### Community 3 - "Tes_YOLO_Image.py"
Cohesion: 0.13
Nodes (23): draw_custom_detections(), get_color_for_class(), load_images_from_path(), main(), on_conf_trackbar(), on_iou_trackbar(), print_detection_summary(), Menampilkan status bar dan HUD informasi di atas gambar. (+15 more)

### Community 4 - "AI_Slat_Rib_Conveyor_V4_2_Robust.py"
Cohesion: 0.16
Nodes (20): analyze_rib_geometry(), analyze_slat_flatness(), _base_component_result(), choose_file_dialog(), clean_path_string(), draw_component_visuals(), extract_clean_contour(), _extract_profile_robust() (+12 more)

### Community 5 - "Tes_Garis_Acuan_PureCV_V4_w_Rib&Slat.py"
Cohesion: 0.23
Nodes (13): Mengosongkan semua state titik ROI dan langkah., Menutup jendela OpenCV dengan tuntas pada sistem operasi Windows., analyze_single_component_pv(), choose_file_dialog(), clean_path_string(), draw_component_visuals(), extract_clean_contour(), get_sibling_images() (+5 more)

### Community 6 - "Benchmark_Filter_Slat_Rib.py"
Cohesion: 0.19
Nodes (10): choose_file_dialog(), clean_path_string(), evaluate_variant(), main(), make_variants(), Benchmark Preprocessing - Seleksi Pipeline & Parameter Filter Paling Efektif…, Membuat 14 varian pipeline preprocessing untuk di-benchmark secara independen.…, Evaluasi kuantitatif kualitas hasil filter untuk metrologi kelurusan: -… (+2 more)

### Community 7 - "Tes_Garis_Acuan_PureCV_V5_w_Rib&Slat_Filter_Benchmark.py"
Cohesion: 0.23
Nodes (13): Membersihkan string path dari quotes, prefix PowerShell, dan spasi., Analisis Metrologi Flatness (P-V) Invarian Sudut: - Garis acuan ideal dihitung…, analyze_single_component_pv(), choose_file_dialog(), clean_path_string(), draw_component_visuals(), extract_clean_contour(), get_sibling_images() (+5 more)

### Community 8 - "Tes_Garis_Acuan_PureCV_V7_w_Rib&Slat_Filter_Benchmark.py"
Cohesion: 0.25
Nodes (11): Membuka dialog File Explorer Windows untuk memilih foto., analyze_single_component_pv(), choose_file_dialog(), clean_path_string(), draw_component_visuals(), extract_clean_contour(), get_sibling_images(), interactive_multi_roi_session() (+3 more)

### Community 9 - "Tes_Garis_Acuan_PureCV_V6_w_Rib&Slat_Filter_Benchmark.py"
Cohesion: 0.28
Nodes (11): Mendapatkan daftar foto yang ada di dalam folder yang sama untuk navigasi cepat., analyze_single_component_pv(), choose_file_dialog(), clean_path_string(), draw_component_visuals(), extract_clean_contour(), get_sibling_images(), interactive_multi_roi_session() (+3 more)

### Community 10 - "Menggambar garis acuan, kontur deteksi, dan vektor lendutan terbesar."
Cohesion: 0.29
Nodes (8): Menggambar garis acuan, kontur deteksi, dan vektor lendutan terbesar., analyze_single_component_pv(), draw_component_visuals(), extract_clean_contour(), interactive_multi_roi_session(), mouse_callback(), Analisis Flatness Mandiri (Peak-to-Valley Metrology V3.5): - Garis acuan ideal…, Ekstraksi kontur fisik tepi dengan 5 tahap Preprocessing + Smart Corner Trimming

### Community 11 - "Tes_Garis_Acuan_PureCV_V1.py"
Cohesion: 0.60
Nodes (4): analyze_slat_polygon(), interactive_point_by_point_session(), mouse_callback(), Analisis kelengkungan plat di dalam poligon yang dipilih titik demi titik.

### Community 12 - "Tes_Garis_Acuan_PureCV_V2.py"
Cohesion: 0.60
Nodes (4): analyze_slat_polygon(), interactive_point_by_point_session(), mouse_callback(), Analisis kelengkungan plat dengan 5 tahap Preprocessing Canggih (V2): 1. CLAHE…

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Profil SIRIP: topmost white pixel dari Morph Close. Algoritma asli…` connect `AI_Slat_Rib_Conveyor_V4_3_Robust_Profile.py` to `Program AI/AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py`, `AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Why does `Menggambar garis acuan, kontur deteksi, dan vektor lendutan terbesar.` connect `Menggambar garis acuan, kontur deteksi, dan vektor lendutan terbesar.` to `AI_Slat_Rib_Conveyor_V4_2_Robust.py`, `Tes_Garis_Acuan_PureCV_V4_w_Rib&Slat.py`, `Tes_Garis_Acuan_PureCV_V5_w_Rib&Slat_Filter_Benchmark.py`, `Tes_Garis_Acuan_PureCV_V7_w_Rib&Slat_Filter_Benchmark.py`, `Tes_Garis_Acuan_PureCV_V6_w_Rib&Slat_Filter_Benchmark.py`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Why does `extract_clean_contour()` connect `Program AI/AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py` to `AI_Slat_Rib_Conveyor_V4_3_Robust_Profile.py`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Should `Program AI/AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09747899159663866 - nodes in this community are weakly interconnected._
- **Should `AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py` be split into smaller, more focused modules?**
  _Cohesion score 0.12315270935960591 - nodes in this community are weakly interconnected._
- **Should `AI_Slat_Rib_Conveyor_V4_3_Robust_Profile.py` be split into smaller, more focused modules?**
  _Cohesion score 0.12315270935960591 - nodes in this community are weakly interconnected._
- **Should `Tes_YOLO_Image.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13043478260869565 - nodes in this community are weakly interconnected._