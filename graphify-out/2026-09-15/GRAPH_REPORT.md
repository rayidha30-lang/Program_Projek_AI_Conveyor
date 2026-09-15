# Graph Report - Program  (2026-09-15)

## Corpus Check
- 41 files · ~140,275 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 7 file(s) not represented in the graph (top: (none) 3, .ico 1, .css 1)

## Summary
- 747 nodes · 1061 edges · 34 communities (30 shown, 2 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 19 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `61301d61`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

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
- Tes_Garis_Acuan_PureCV_V3_w_Rib&Slat.py
- Tes_Garis_Acuan_PureCV_V1.py
- Tes_Garis_Acuan_PureCV_V2.py
- package.json
- dependencies
- Index.tsx
- AI_Slat_Rib_Conveyor_V4_5_Robust_Profile_W_db.py
- components.json
- compilerOptions
- devDependencies
- AI_Slat_Rib_Conveyor_V4_5_Geometry_Normalized_W_db.py
- 📋 Penjelasan Per Fase
- button.tsx
- eslint.config.js
- scripts
- Welcome to your Lovable project
- vite.config.ts
- rules/graphify.md
- workflows/graphify.md
- AI_Slat_Rib_Conveyor_V4_6_Robust_Profile_W_db_OK.py
- AI_Slat_Rib_Conveyor_V4_7_Robust_Profile_W_db_OK.py
- AI_Slat_Rib_Conveyor_V4_6_Geometry_Normalized_W_db.py

## God Nodes (most connected - your core abstractions)
1. `compilerOptions` - 17 edges
2. `save_inspection_to_db()` - 17 edges
3. `interactive_multi_roi_session()` - 15 edges
4. `interactive_multi_roi_session()` - 15 edges
5. `interactive_multi_roi_session()` - 15 edges
6. `interactive_multi_roi_session()` - 14 edges
7. `interactive_multi_roi_session()` - 14 edges
8. `interactive_multi_roi_session()` - 14 edges
9. `interactive_multi_roi_session()` - 12 edges
10. `interactive_multi_roi_session()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `interactive_multi_roi_session()` --calls--> `save_inspection_to_db()`  [EXTRACTED]
  Program AI/AI_Slat_Rib_Conveyor_V4_5_Robust_Profile_W_db.py → Program AI/db.py
- `interactive_multi_roi_session()` --calls--> `save_inspection_to_db()`  [EXTRACTED]
  Program AI/AI_Slat_Rib_Conveyor_V4_6_Geometry_Normalized_W_db.py → Program AI/db.py
- `interactive_multi_roi_session()` --calls--> `save_inspection_to_db()`  [EXTRACTED]
  Program AI/AI_Slat_Rib_Conveyor_V4_6_Robust_Profile_W_db_OK.py → Program AI/db.py
- `interactive_multi_roi_session()` --calls--> `save_inspection_to_db()`  [EXTRACTED]
  Program AI/AI_Slat_Rib_Conveyor_V4_7_Robust_Profile_W_db_OK.py → Program AI/db.py
- `Button` --calls--> `cn()`  [EXTRACTED]
  FE/src/components/ui/button.tsx → FE/src/lib/utils.ts

## Import Cycles
- None detected.

## Communities (34 total, 2 thin omitted)

### Community 0 - "Program AI/AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py"
Cohesion: 0.09
Nodes (35): analyze_sirip_pv(), analyze_slat_profile(), choose_file_dialog(), clean_path_string(), _ctx_bounds(), draw_sirip_visuals(), draw_slat_visuals(), extract_clean_contour() (+27 more)

### Community 1 - "AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py"
Cohesion: 0.10
Nodes (31): analyze_sirip_pv(), analyze_slat_profile(), choose_file_dialog(), clean_path_string(), _ctx_bounds(), draw_sirip_visuals(), draw_slat_visuals(), extract_clean_contour() (+23 more)

### Community 2 - "AI_Slat_Rib_Conveyor_V4_3_Robust_Profile.py"
Cohesion: 0.10
Nodes (31): analyze_sirip_pv(), analyze_slat_profile(), choose_file_dialog(), clean_path_string(), _ctx_bounds(), draw_sirip_visuals(), draw_slat_visuals(), extract_clean_contour() (+23 more)

### Community 3 - "Tes_YOLO_Image.py"
Cohesion: 0.13
Nodes (23): draw_custom_detections(), get_color_for_class(), load_images_from_path(), main(), on_conf_trackbar(), on_iou_trackbar(), print_detection_summary(), Menampilkan status bar dan HUD informasi di atas gambar. (+15 more)

### Community 4 - "AI_Slat_Rib_Conveyor_V4_2_Robust.py"
Cohesion: 0.12
Nodes (25): analyze_rib_geometry(), analyze_slat_flatness(), _base_component_result(), choose_file_dialog(), clean_path_string(), draw_component_visuals(), extract_clean_contour(), _extract_profile_robust() (+17 more)

### Community 5 - "Tes_Garis_Acuan_PureCV_V4_w_Rib&Slat.py"
Cohesion: 0.15
Nodes (18): analyze_single_component_pv(), choose_file_dialog(), clean_path_string(), draw_component_visuals(), extract_clean_contour(), get_sibling_images(), interactive_multi_roi_session(), mouse_callback() (+10 more)

### Community 6 - "Benchmark_Filter_Slat_Rib.py"
Cohesion: 0.11
Nodes (22): build_top4_dashboard(), choose_file_dialog(), clean_path_string(), export_presets_to_code(), extract_and_evaluate_profile(), get_sibling_images(), main(), Benchmark Preprocessing & Seleksi Filter Adaptif Multi-Environment untuk… (+14 more)

### Community 7 - "Tes_Garis_Acuan_PureCV_V5_w_Rib&Slat_Filter_Benchmark.py"
Cohesion: 0.15
Nodes (18): analyze_single_component_pv(), choose_file_dialog(), clean_path_string(), draw_component_visuals(), extract_clean_contour(), get_sibling_images(), interactive_multi_roi_session(), mouse_callback() (+10 more)

### Community 8 - "Tes_Garis_Acuan_PureCV_V7_w_Rib&Slat_Filter_Benchmark.py"
Cohesion: 0.14
Nodes (18): analyze_single_component_pv(), choose_file_dialog(), clean_path_string(), draw_component_visuals(), extract_clean_contour(), get_sibling_images(), interactive_multi_roi_session(), mouse_callback() (+10 more)

### Community 9 - "Tes_Garis_Acuan_PureCV_V6_w_Rib&Slat_Filter_Benchmark.py"
Cohesion: 0.15
Nodes (18): analyze_single_component_pv(), choose_file_dialog(), clean_path_string(), draw_component_visuals(), extract_clean_contour(), get_sibling_images(), interactive_multi_roi_session(), mouse_callback() (+10 more)

### Community 10 - "Tes_Garis_Acuan_PureCV_V3_w_Rib&Slat.py"
Cohesion: 0.29
Nodes (8): analyze_single_component_pv(), draw_component_visuals(), extract_clean_contour(), interactive_multi_roi_session(), mouse_callback(), Analisis Flatness Mandiri (Peak-to-Valley Metrology V3.5): - Garis acuan ideal…, Menggambar garis acuan, kontur deteksi, dan vektor lendutan terbesar, Ekstraksi kontur fisik tepi dengan 5 tahap Preprocessing + Smart Corner Trimming

### Community 11 - "Tes_Garis_Acuan_PureCV_V1.py"
Cohesion: 0.60
Nodes (4): analyze_slat_polygon(), interactive_point_by_point_session(), mouse_callback(), Analisis kelengkungan plat di dalam poligon yang dipilih titik demi titik.

### Community 12 - "Tes_Garis_Acuan_PureCV_V2.py"
Cohesion: 0.60
Nodes (4): analyze_slat_polygon(), interactive_point_by_point_session(), mouse_callback(), Analisis kelengkungan plat dengan 5 tahap Preprocessing Canggih (V2): 1. CLAHE…

### Community 14 - "package.json"
Cohesion: 0.04
Nodes (52): name, private, type, version, cmdk, date-fns, embla-carousel-react, eslint (+44 more)

### Community 15 - "dependencies"
Cohesion: 0.04
Nodes (50): dependencies, class-variance-authority, clsx, cmdk, date-fns, embla-carousel-react, @hookform/resolvers, input-otp (+42 more)

### Community 16 - "Index.tsx"
Cohesion: 0.06
Nodes (29): App(), queryClient, AbnormalComponentSummary, AnalyticsData, API_BASE_URL, EvidenceItem, fetchAnalytics(), fetchEvidence() (+21 more)

### Community 17 - "AI_Slat_Rib_Conveyor_V4_5_Robust_Profile_W_db.py"
Cohesion: 0.09
Nodes (35): analyze_sirip_pv(), analyze_slat_profile(), choose_file_dialog(), clean_path_string(), _ctx_bounds(), draw_sirip_visuals(), draw_slat_visuals(), extract_clean_contour() (+27 more)

### Community 18 - "components.json"
Cohesion: 0.11
Nodes (18): aliases, components, hooks, lib, ui, utils, iconLibrary, registries (+10 more)

### Community 19 - "compilerOptions"
Cohesion: 0.11
Nodes (18): compilerOptions, baseUrl, jsx, lib, module, moduleResolution, noEmit, noFallthroughCasesInSwitch (+10 more)

### Community 20 - "devDependencies"
Cohesion: 0.12
Nodes (16): devDependencies, eslint, eslint-config-prettier, @eslint/js, eslint-plugin-prettier, eslint-plugin-react-hooks, eslint-plugin-react-refresh, globals (+8 more)

### Community 21 - "AI_Slat_Rib_Conveyor_V4_5_Geometry_Normalized_W_db.py"
Cohesion: 0.07
Nodes (42): choose_file_dialog(), clean_path_string(), draw_workpiece_visuals(), extract_canonical_profile_workpiece(), extract_clean_contour_fallback(), get_sibling_images(), interactive_multi_roi_session(), mouse_callback() (+34 more)

### Community 22 - "📋 Penjelasan Per Fase"
Cohesion: 0.17
Nodes (11): Alur Kerja Sistem Inspeksi Slat & Rib Conveyor, 🗺️ Diagram Alur Lengkap, FASE 1 — Persiapan ROI, FASE 2 — Pre-Processing Filter, FASE 3 — Ekstraksi Kontur → 🔵 Garis Biru, FASE 4 — Garis Acuan Ideal → 🟡 Garis Kuning, FASE 5 — Pengukuran Deviasi → 🔴 Garis Merah, FASE 6 — Penilaian OK / NG (+3 more)

### Community 23 - "button.tsx"
Cohesion: 0.27
Nodes (8): Button, ButtonProps, buttonVariants, cn(), class-variance-authority, clsx, @radix-ui/react-slot, tailwind-merge

### Community 24 - "eslint.config.js"
Cohesion: 0.33
Nodes (5): @eslint/js, eslint-plugin-react-hooks, eslint-plugin-react-refresh, globals, typescript-eslint

### Community 25 - "scripts"
Cohesion: 0.33
Nodes (6): scripts, build, dev, format, lint, preview

### Community 26 - "Welcome to your Lovable project"
Cohesion: 0.40
Nodes (4): Build with Lovable, Built with, Development, Welcome to your Lovable project

### Community 27 - "vite.config.ts"
Cohesion: 0.50
Nodes (3): @tailwindcss/vite, vite, @vitejs/plugin-react

### Community 31 - "AI_Slat_Rib_Conveyor_V4_6_Robust_Profile_W_db_OK.py"
Cohesion: 0.09
Nodes (35): analyze_sirip_pv(), analyze_slat_profile(), choose_file_dialog(), clean_path_string(), _ctx_bounds(), draw_sirip_visuals(), draw_slat_visuals(), extract_clean_contour() (+27 more)

### Community 32 - "AI_Slat_Rib_Conveyor_V4_7_Robust_Profile_W_db_OK.py"
Cohesion: 0.14
Nodes (16): choose_file_dialog(), clean_path_string(), draw_workpiece_visuals(), extract_canonical_profile_workpiece(), extract_clean_contour_fallback(), get_sibling_images(), interactive_multi_roi_session(), mouse_callback() (+8 more)

### Community 33 - "AI_Slat_Rib_Conveyor_V4_6_Geometry_Normalized_W_db.py"
Cohesion: 0.05
Nodes (56): analyze_sirip_pv(), analyze_slat_profile(), _candidate_support(), _candidate_vertical_support(), choose_file_dialog(), clean_path_string(), _ctx_bounds(), draw_sirip_visuals() (+48 more)

## Knowledge Gaps
- **177 isolated node(s):** `$schema`, `style`, `rsc`, `tsx`, `css` (+172 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 393 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `save_inspection_to_db()` connect `AI_Slat_Rib_Conveyor_V4_5_Geometry_Normalized_W_db.py` to `AI_Slat_Rib_Conveyor_V4_7_Robust_Profile_W_db_OK.py`, `AI_Slat_Rib_Conveyor_V4_5_Robust_Profile_W_db.py`, `AI_Slat_Rib_Conveyor_V4_6_Robust_Profile_W_db_OK.py`, `AI_Slat_Rib_Conveyor_V4_6_Geometry_Normalized_W_db.py`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Why does `dependencies` connect `dependencies` to `package.json`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Why does `devDependencies` connect `devDependencies` to `package.json`?**
  _High betweenness centrality (0.010) - this node is a cross-community bridge._
- **What connects `$schema`, `style`, `rsc` to the rest of the system?**
  _177 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Program AI/AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08677098150782361 - nodes in this community are weakly interconnected._
- **Should `AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09803921568627451 - nodes in this community are weakly interconnected._
- **Should `AI_Slat_Rib_Conveyor_V4_3_Robust_Profile.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09803921568627451 - nodes in this community are weakly interconnected._