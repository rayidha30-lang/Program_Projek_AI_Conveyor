import React, { useState, useEffect } from "react";
import {
  Check,
  RotateCcw,
  Wrench,
  AlertTriangle,
  Layers,
  ShieldCheck,
  Info,
  Clock3,
  Sliders,
  Sparkles,
  RefreshCw,
  Activity,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  fetchTopInspectionConfig,
  updateTopInspectionConfig,
  SCALE_FACTOR_MM_PER_PX,
  type TopInspectionThresholds,
} from "@/lib/api";

// Nilai Default Emas (Baseline Aman Teruji)
const GOLDEN_DEFAULTS: TopInspectionThresholds = {
  rib_tip_threshold: 9.5,
  rib_dev_threshold: 8.5,
  rib_pv_threshold: 14.0,
  rib_sagitta_threshold: 11.5,
  rib_conf_threshold: 0.55,
  max_flatness_pv: 16.5,
  max_sagitta: 8.6,
  max_center_void: 1450,
  min_continuity_cov: 0.25,
  slat_max_gap_limit: 45.0,
  slat_delta_limit: 14.0,
  slat_valley_sens: 5.0,
};

interface TuningSettingsProps {
  theme?: "light" | "dark";
}

export default function TuningSettings({ theme = "dark" }: TuningSettingsProps) {
  const isDark = theme === "dark";
  const [activeCategory, setActiveCategory] = useState<"all" | "rib" | "slat">("all");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>("");

  const [thresholds, setThresholds] = useState<TopInspectionThresholds>(GOLDEN_DEFAULTS);
  const [originalThresholds, setOriginalThresholds] = useState<TopInspectionThresholds>(GOLDEN_DEFAULTS);

  // Ambil konfigurasi dari Flask API saat mount
  const loadConfig = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const res = await fetchTopInspectionConfig();
      if (res && res.config && res.config.thresholds) {
        setThresholds({ ...GOLDEN_DEFAULTS, ...res.config.thresholds });
        setOriginalThresholds({ ...GOLDEN_DEFAULTS, ...res.config.thresholds });
        if (res.config.updated_at) {
          setLastUpdated(res.config.updated_at);
        }
      }
    } catch (err: any) {
      console.warn("Gagal memuat konfigurasi dari API, menggunakan default emas:", err);
      setErrorMessage("Koneksi API offline, menampilkan parameter tersimpan lokal.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadConfig();
  }, []);

  const handleChange = (key: keyof TopInspectionThresholds, value: number) => {
    setThresholds((prev) => ({
      ...prev,
      [key]: value,
    }));
    setSaveSuccess(false);
  };

  const handleResetToGolden = () => {
    setThresholds({ ...GOLDEN_DEFAULTS });
    setSaveSuccess(false);
  };

  const handleSave = async () => {
    setSaving(true);
    setErrorMessage(null);
    try {
      const res = await updateTopInspectionConfig(thresholds);
      setOriginalThresholds({ ...thresholds });
      setLastUpdated(res.updated_at || new Date().toLocaleTimeString());
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 4000);
    } catch (err: any) {
      setErrorMessage(err.message || "Gagal menyimpan konfigurasi.");
    } finally {
      setSaving(false);
    }
  };

  const hasChanges = JSON.stringify(thresholds) !== JSON.stringify(originalThresholds);

  return (
    <div className="space-y-6">
      {/* Header Banner & Status Bar */}
      <div
        className={`relative overflow-hidden rounded-2xl border p-5 shadow-sm transition-all ${
          isDark
            ? "border-emerald-500/20 bg-linear-to-r from-slate-900 via-slate-900/90 to-emerald-950/40 text-white"
            : "border-purple-200 bg-linear-to-r from-purple-50 via-white to-indigo-50 text-slate-900"
        }`}
      >
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="flex size-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="font-mono text-[11px] font-bold tracking-wider uppercase text-emerald-400">
              Auto Hot-Reload Engine Aktif
            </span>
            <span className="rounded-md bg-emerald-500/10 px-2 py-0.5 font-mono text-[10px] font-bold text-emerald-400 border border-emerald-500/20">
              Zero Downtime
            </span>
            <span className="rounded-md bg-sky-500/10 px-2 py-0.5 font-mono text-[10px] font-bold text-sky-400 border border-sky-500/20">
              Skala Metrologi: 0.515 mm/px
            </span>
            <span className="text-[11px] text-muted-foreground hidden sm:inline">
              • Parameter langsung diterapkan ke kamera secara seketika tanpa restart
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <Button
              variant="outline"
              size="sm"
              onClick={handleResetToGolden}
              disabled={saving}
              className={`text-xs font-semibold cursor-pointer ${
                isDark ? "border-slate-700 hover:bg-slate-800" : "border-slate-300"
              }`}
            >
              <RotateCcw className="mr-1.5 size-3.5" />
              Reset ke Default
            </Button>
            <Button
              size="sm"
              onClick={handleSave}
              disabled={saving}
              className={`text-xs font-bold text-slate-950 transition-all cursor-pointer ${
                saveSuccess
                  ? "bg-emerald-400 hover:bg-emerald-500"
                  : "bg-[#00E599] hover:bg-[#00d089] shadow-lg shadow-emerald-500/25"
              }`}
            >
              {saving ? (
                <>
                  <RefreshCw className="mr-1.5 size-3.5 animate-spin" />
                  Menerapkan...
                </>
              ) : saveSuccess ? (
                <>
                  <Check className="mr-1.5 size-3.5" />
                  Tersimpan & Aktif!
                </>
              ) : (
                <>
                  <Wrench className="mr-1.5 size-3.5" />
                  Simpan & Terapkan
                </>
              )}
            </Button>
          </div>
        </div>

        {/* Status Toast / Alert Bar */}
        {(saveSuccess || errorMessage || lastUpdated) && (
          <div className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-border/40 pt-3 text-[11px]">
            <div className="flex items-center gap-2">
              {saveSuccess && (
                <span className="flex items-center gap-1 font-bold text-emerald-400">
                  <Check className="size-3.5" />
                  Pengaturan baru berhasil dikirim dan diterapkan ke Jetson Vision Engine!
                </span>
              )}
              {errorMessage && (
                <span className="flex items-center gap-1 font-bold text-rose-400">
                  <AlertTriangle className="size-3.5" />
                  {errorMessage}
                </span>
              )}
              {!saveSuccess && !errorMessage && hasChanges && (
                <span className="flex items-center gap-1 font-semibold text-amber-400">
                  <Sparkles className="size-3.5" />
                  Ada perubahan parameter yang belum disimpan.
                </span>
              )}
            </div>
            {lastUpdated && (
              <span className="flex items-center gap-1 font-mono text-[10px] text-muted-foreground">
                <Clock3 className="size-3" />
                Sinkronisasi Terakhir: {lastUpdated}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Filter Kategori Tab & Petunjuk Indikator Pembeda */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-3">
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => setActiveCategory("all")}
            className={`flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-xs font-bold transition-all cursor-pointer ${
              activeCategory === "all"
                ? isDark
                  ? "bg-slate-100 text-slate-950 shadow-md font-extrabold"
                  : "bg-slate-900 text-white shadow-md font-extrabold"
                : "text-muted-foreground hover:bg-muted/60"
            }`}
          >
            <Sliders className="size-3.5" />
            <span>Semua Parameter</span>
            <span
              className={`rounded-full px-1.5 py-0.5 text-[10px] font-mono font-bold ${
                activeCategory === "all"
                  ? isDark
                    ? "bg-slate-900/20 text-slate-950"
                    : "bg-white/30 text-white"
                  : "bg-muted text-muted-foreground"
              }`}
            >
              8
            </span>
          </button>
          <button
            onClick={() => setActiveCategory("rib")}
            className={`flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-xs font-bold transition-all cursor-pointer ${
              activeCategory === "rib"
                ? "bg-emerald-500 text-slate-950 shadow-md font-extrabold"
                : "text-muted-foreground hover:bg-emerald-500/10 hover:text-emerald-400"
            }`}
          >
            <Activity className="size-3.5" />
            <span>Bilah Sirip (Rib)</span>
            <span
              className={`rounded-full px-1.5 py-0.5 text-[10px] font-mono font-bold ${
                activeCategory === "rib"
                  ? "bg-slate-950/25 text-slate-950"
                  : "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
              }`}
            >
              5
            </span>
          </button>
          <button
            onClick={() => setActiveCategory("slat")}
            className={`flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-xs font-bold transition-all cursor-pointer ${
              activeCategory === "slat"
                ? "bg-sky-500 text-slate-950 shadow-md font-extrabold"
                : "text-muted-foreground hover:bg-sky-500/10 hover:text-sky-400"
            }`}
          >
            <Layers className="size-3.5" />
            <span>Plat Konveyor (Slat)</span>
            <span
              className={`rounded-full px-1.5 py-0.5 text-[10px] font-mono font-bold ${
                activeCategory === "slat"
                  ? "bg-slate-950/25 text-slate-950"
                  : "bg-sky-500/15 text-sky-400 border border-sky-500/30"
              }`}
            >
              3
            </span>
          </button>
        </div>

        {/* Legend Pembeda Visual Warna */}
        <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <span className="size-2.5 rounded-full bg-emerald-400 shadow-xs shadow-emerald-400/50" />
            <span className="font-semibold text-emerald-400">Emerald:</span>
            <span>Bilah Sirip ({activeCategory === "slat" ? "Disembunyikan" : "5"})</span>
          </span>
          <span className="text-border">|</span>
          <span className="flex items-center gap-1.5">
            <span className="size-2.5 rounded-full bg-sky-400 shadow-xs shadow-sky-400/50" />
            <span className="font-semibold text-sky-400">Sky Blue:</span>
            <span>Plat Konveyor ({activeCategory === "rib" ? "Disembunyikan" : "3"})</span>
          </span>
        </div>
      </div>

      {/* Wadah Seksi Terkelompok */}
      <div className="space-y-8">
        {/* ========================================================
            BAGIAN 1: PARAMETER BILAH SIRIP (RIB)
           ======================================================== */}
        {(activeCategory === "all" || activeCategory === "rib") && (
          <div className="space-y-4">
            {/* Header Seksi Bilah Sirip */}
            <div
              className={`flex flex-wrap items-center justify-between gap-3 rounded-2xl border p-4 shadow-xs transition-all ${
                isDark
                  ? "border-emerald-500/30 bg-linear-to-r from-emerald-950/40 via-slate-900/90 to-slate-900 text-white"
                  : "border-emerald-200 bg-linear-to-r from-emerald-50 via-white to-emerald-50/40 text-slate-900"
              }`}
            >
              <div className="flex items-center gap-3">
                <div
                  className={`flex size-10 items-center justify-center rounded-xl border shadow-sm ${
                    isDark
                      ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-400"
                      : "border-emerald-300 bg-emerald-100 text-emerald-700"
                  }`}
                >
                  <Activity className="size-5" />
                </div>
                <div>
                  <div className="flex items-center gap-2.5">
                    <h3 className="text-sm font-black tracking-tight sm:text-base">
                      Kelompok Parameter Bilah Sirip (Rib)
                    </h3>
                    <span
                      className={`rounded-full px-2 py-0.5 font-mono text-[10px] font-extrabold border ${
                        isDark
                          ? "border-emerald-500/40 bg-emerald-500/20 text-emerald-300"
                          : "border-emerald-300 bg-emerald-100 text-emerald-800"
                      }`}
                    >
                      5 Parameter
                    </span>
                  </div>
                  <p className="text-[11px] text-muted-foreground mt-0.5">
                    Mengevaluasi kelurusan kontur, simpangan lekukan bodi bilah sirip, toleransi ujung, dan ambang deteksi AI YOLO.
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <span
                  className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1 font-mono text-[10.5px] font-bold border ${
                    isDark
                      ? "border-emerald-500/30 bg-slate-900/80 text-emerald-400"
                      : "border-emerald-200 bg-white text-emerald-700 shadow-xs"
                  }`}
                >
                  <span className="size-2 rounded-full bg-emerald-400 animate-pulse" />
                  Target: CAM-01 • Bilah Sirip (Rib)
                </span>
              </div>
            </div>

            {/* Grid Kartu Rib */}
            <div className="grid gap-6 md:grid-cols-2">
              <ParameterCard
                category="rib"
                title="Toleransi Ujung Sirip (Tip Threshold)"
                paramKey="rib_tip_threshold"
                unit="px"
                value={thresholds.rib_tip_threshold}
                min={4.0}
                max={25.0}
                step={0.5}
                defaultValue={GOLDEN_DEFAULTS.rib_tip_threshold}
                onChange={(v) => handleChange("rib_tip_threshold", v)}
                isDark={isDark}
                notice="Mengatur batas kemiringan atau lekukan di ujung bilah sirip (ujung kiri dan ujung kanan). Jika ujung sirip terangkat naik atau menukik turun melebihi angka ini dari garis lurus, sistem akan menandainya sebagai NG. Naikkan angka ini jika ujung sirip di konveyor aslinya memang memiliki potongan sudut miring agar tidak salah terbaca NG."
                diagram={<TipDiagram value={thresholds.rib_tip_threshold} isDark={isDark} />}
              />

              <ParameterCard
                category="rib"
                title="Toleransi Badan Sirip (Body Deviation)"
                paramKey="rib_dev_threshold"
                unit="px"
                value={thresholds.rib_dev_threshold}
                min={3.0}
                max={20.0}
                step={0.5}
                defaultValue={GOLDEN_DEFAULTS.rib_dev_threshold}
                onChange={(v) => handleChange("rib_dev_threshold", v)}
                isDark={isDark}
                notice="Mengatur batas kelurusan di sepanjang badan tengah sirip. Jika bagian tengah sirip bergelombang atau meliuk melebihi angka ini terhadap garis lurus acuan, sirip akan dianggap NG."
                diagram={<BodyDevDiagram value={thresholds.rib_dev_threshold} isDark={isDark} />}
              />

              <ParameterCard
                category="rib"
                title="Flatness Peak-to-Valley Sirip (PV)"
                paramKey="rib_pv_threshold"
                unit="px"
                value={thresholds.rib_pv_threshold}
                min={5.0}
                max={30.0}
                step={0.5}
                defaultValue={GOLDEN_DEFAULTS.rib_pv_threshold}
                onChange={(v) => handleChange("rib_pv_threshold", v)}
                isDark={isDark}
                notice="Mengukur selisih ketinggian antara titik paling tinggi dan titik paling rendah di sepanjang bilah sirip. Semakin kecil nilainya, sirip dituntut harus semakin rata dan tegak lurus sempurna."
                diagram={<FlatnessPVDiagram value={thresholds.rib_pv_threshold} isDark={isDark} />}
              />

              <ParameterCard
                category="rib"
                title="Kelengkungan Busur Sirip (Sagitta)"
                paramKey="rib_sagitta_threshold"
                unit="px"
                value={thresholds.rib_sagitta_threshold}
                min={4.0}
                max={25.0}
                step={0.5}
                defaultValue={GOLDEN_DEFAULTS.rib_sagitta_threshold}
                onChange={(v) => handleChange("rib_sagitta_threshold", v)}
                isDark={isDark}
                notice="Mengukur kelengkungan badan sirip yang melengkung menyerupai busur panah. Jika bagian tengah sirip melengkung cembung atau cekung melebihi angka ini, sistem akan menandainya sebagai NG."
                diagram={<SagittaDiagram value={thresholds.rib_sagitta_threshold} isDark={isDark} />}
              />

              <ParameterCard
                category="rib"
                title="Ambang Keyakinan AI YOLO Sirip"
                paramKey="rib_conf_threshold"
                unit="ratio"
                value={thresholds.rib_conf_threshold}
                min={0.3}
                max={0.9}
                step={0.05}
                defaultValue={GOLDEN_DEFAULTS.rib_conf_threshold}
                onChange={(v) => handleChange("rib_conf_threshold", v)}
                isDark={isDark}
                displayTransform={(v) => `${Math.round(v * 100)}%`}
                notice="Tingkat kepekaan kamera dan model AI dalam mengenali bilah sirip. Nilai standar 55% memastikan sistem hanya memeriksa objek yang benar-benar sirip, bukan bayangan atau barang lain."
                diagram={<ConfidenceDiagram value={thresholds.rib_conf_threshold} isDark={isDark} />}
              />
            </div>
          </div>
        )}

        {/* ========================================================
            BAGIAN 2: PARAMETER PLAT KONVEYOR & CELAH NAT (SLAT)
           ======================================================== */}
        {(activeCategory === "all" || activeCategory === "slat") && (
          <div className="space-y-4">
            {/* Header Seksi Plat Slat */}
            <div
              className={`flex flex-wrap items-center justify-between gap-3 rounded-2xl border p-4 shadow-xs transition-all ${
                isDark
                  ? "border-sky-500/30 bg-linear-to-r from-sky-950/40 via-slate-900/90 to-slate-900 text-white"
                  : "border-sky-200 bg-linear-to-r from-sky-50 via-white to-sky-50/40 text-slate-900"
              }`}
            >
              <div className="flex items-center gap-3">
                <div
                  className={`flex size-10 items-center justify-center rounded-xl border shadow-sm ${
                    isDark
                      ? "border-sky-500/40 bg-sky-500/15 text-sky-400"
                      : "border-sky-300 bg-sky-100 text-sky-700"
                  }`}
                >
                  <Layers className="size-5" />
                </div>
                <div>
                  <div className="flex items-center gap-2.5">
                    <h3 className="text-sm font-black tracking-tight sm:text-base">
                      Kelompok Parameter Plat Konveyor (Slat)
                    </h3>
                    <span
                      className={`rounded-full px-2 py-0.5 font-mono text-[10px] font-extrabold border ${
                        isDark
                          ? "border-sky-500/40 bg-sky-500/20 text-sky-300"
                          : "border-sky-300 bg-sky-100 text-sky-800"
                      }`}
                    >
                      3 Parameter
                    </span>
                  </div>
                  <p className="text-[11px] text-muted-foreground mt-0.5">
                    Mengevaluasi kelurusan celah nat sambungan plat, lengkungan lembaran plat slat, dan toleransi bukaan celah tengah.
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <span
                  className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1 font-mono text-[10.5px] font-bold border ${
                    isDark
                      ? "border-sky-500/30 bg-slate-900/80 text-sky-400"
                      : "border-sky-200 bg-white text-sky-700 shadow-xs"
                  }`}
                >
                  <span className="size-2 rounded-full bg-sky-400 animate-pulse" />
                  Target: CAM-01 • Plat Slat & Celah Sambungan
                </span>
              </div>
            </div>

            {/* Grid Kartu Slat */}
            <div className="grid gap-6 md:grid-cols-2">
              <ParameterCard
                category="slat"
                title="Flatness PV Celah Plat Slat"
                paramKey="max_flatness_pv"
                unit="px"
                value={thresholds.max_flatness_pv}
                min={5.0}
                max={35.0}
                step={0.5}
                defaultValue={GOLDEN_DEFAULTS.max_flatness_pv}
                onChange={(v) => handleChange("max_flatness_pv", v)}
                isDark={isDark}
                notice="Mengatur batas kerataan celah sambungan plat slat. Jika plat penyok atau bergelombang sehingga garis sambungannya tidak rata melebihi nilai ini, plat akan dinilai NG."
                diagram={<SlatFlatnessDiagram value={thresholds.max_flatness_pv} isDark={isDark} />}
              />

              <ParameterCard
                category="slat"
                title="Kelengkungan Busur Plat Slat (Sagitta)"
                paramKey="max_sagitta"
                unit="px"
                value={thresholds.max_sagitta}
                min={3.0}
                max={20.0}
                step={0.2}
                defaultValue={GOLDEN_DEFAULTS.max_sagitta}
                onChange={(v) => handleChange("max_sagitta", v)}
                isDark={isDark}
                notice="Menilai apakah lembaran plat konveyor melengkung di bagian tengah akibat beban kerja yang berat."
                diagram={<SlatSagittaDiagram value={thresholds.max_sagitta} isDark={isDark} />}
              />

              <ParameterCard
                category="slat"
                title="Toleransi Celah Kosong Sambungan (Center Void)"
                paramKey="max_center_void"
                unit="px"
                value={thresholds.max_center_void}
                min={500}
                max={3000}
                step={50}
                defaultValue={GOLDEN_DEFAULTS.max_center_void}
                onChange={(v) => handleChange("max_center_void", v)}
                isDark={isDark}
                notice="Mengatur batas renggang celah sambungan antar plat di bagian tengah konveyor. Jika celah plat terlalu lebar atau terputus melebihi batas ini, sistem akan menandainya."
                diagram={<VoidGapDiagram value={thresholds.max_center_void} isDark={isDark} />}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// =========================================================================
// KOMPONEN KARTU PARAMETER REUSABLE
// =========================================================================
interface ParameterCardProps {
  category: "rib" | "slat";
  title: string;
  paramKey: string;
  unit: string;
  value: number;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
  onChange: (val: number) => void;
  isDark: boolean;
  notice: string;
  diagram: React.ReactNode;
  displayTransform?: (v: number) => string;
  scaleFactor?: number;
}

function ParameterCard({
  category,
  title,
  paramKey,
  unit,
  value,
  min,
  max,
  step,
  defaultValue,
  onChange,
  isDark,
  notice,
  diagram,
  displayTransform,
  scaleFactor = SCALE_FACTOR_MM_PER_PX,
}: ParameterCardProps) {
  const isModified = value !== defaultValue;
  const isRib = category === "rib";

  const formatValue = (v: number) => {
    if (displayTransform) return displayTransform(v);
    if (unit === "px") {
      const mm = v * (scaleFactor || 0.515);
      const pxStr = Number.isInteger(v) ? `${v} px` : `${v.toFixed(1)} px`;
      return `${mm.toFixed(1)} mm (${pxStr})`;
    }
    return `${v} ${unit}`;
  };

  return (
    <div
      className={`relative flex flex-col justify-between rounded-xl border border-l-4 p-4.5 shadow-sm transition-all ${
        isRib
          ? isDark
            ? "border-border/60 border-l-emerald-400 bg-card/95 hover:border-emerald-500/40"
            : "border-border/80 border-l-emerald-500 bg-white hover:border-emerald-300"
          : isDark
          ? "border-border/60 border-l-sky-400 bg-card/95 hover:border-sky-500/40"
          : "border-border/80 border-l-sky-500 bg-white hover:border-sky-300"
      }`}
    >
      <div>
        {/* Category Pill Badge & Metadata Header */}
        <div className="mb-2.5 flex items-center justify-between gap-2">
          {isRib ? (
            <span
              className={`inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[10px] font-extrabold uppercase tracking-wide border ${
                isDark
                  ? "border-emerald-500/30 bg-emerald-500/15 text-emerald-400"
                  : "border-emerald-300 bg-emerald-50 text-emerald-800"
              }`}
            >
              <Activity className="size-3" />
              Bilah Sirip (Rib)
            </span>
          ) : (
            <span
              className={`inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[10px] font-extrabold uppercase tracking-wide border ${
                isDark
                  ? "border-sky-500/30 bg-sky-500/15 text-sky-400"
                  : "border-sky-300 bg-sky-50 text-sky-800"
              }`}
            >
              <Layers className="size-3" />
              Plat Konveyor (Slat)
            </span>
          )}

          <div className="flex items-center gap-1.5">
            {isModified && (
              <span className="rounded bg-amber-500/10 px-1.5 py-0.5 text-[9px] font-bold text-amber-400 border border-amber-500/20">
                Diubah
              </span>
            )}
            <span className="font-mono text-[9px] text-muted-foreground/80">
              {isRib ? "CAM-01 • Rib" : "CAM-01 • Slat"}
            </span>
          </div>
        </div>

        {/* Header Kartu: Judul Parameter & Nilai Saat Ini */}
        <div className="flex items-start justify-between gap-3">
          <div>
            <h3 className="text-sm font-bold tracking-tight">{title}</h3>
            <span className="font-mono text-[10px] text-muted-foreground">{paramKey}</span>
          </div>
          <div
            className={`shrink-0 rounded-lg px-2.5 py-1 font-mono text-xs font-black shadow-xs border ${
              isRib
                ? isDark
                  ? "bg-emerald-950/40 text-emerald-300 border-emerald-500/30"
                  : "bg-emerald-50 text-emerald-800 border-emerald-200"
                : isDark
                ? "bg-sky-950/40 text-sky-300 border-sky-500/30"
                : "bg-sky-50 text-sky-800 border-sky-200"
            }`}
          >
            {formatValue(value)}
          </div>
        </div>

        {/* Diagram Skematik Visual */}
        <div className="my-3 overflow-hidden rounded-lg border border-border/50 bg-black/25 p-2">
          {diagram}
        </div>

        {/* Kotak Pemberitahuan Edukatif untuk Operator */}
        <div
          className={`flex items-start gap-2 rounded-lg p-2.5 text-[11px] leading-relaxed border ${
            isDark
              ? "bg-slate-800/60 text-slate-300 border-slate-700/50"
              : isRib
              ? "bg-emerald-50/40 text-slate-700 border-emerald-100"
              : "bg-sky-50/40 text-slate-700 border-sky-100"
          }`}
        >
          <Info
            className={`size-4 shrink-0 mt-0.5 ${
              isRib ? "text-emerald-400" : "text-sky-400"
            }`}
          />
          <p>{notice}</p>
        </div>
      </div>

      {/* Kontrol Slider & Input */}
      <div className="mt-4 pt-3 border-t border-border/40">
        <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-1.5">
          <span>Min: {formatValue(min)}</span>
          <span className="font-semibold text-foreground">
            Default: {formatValue(defaultValue)}
          </span>
          <span>Maks: {formatValue(max)}</span>
        </div>
        <div className="flex items-center gap-3">
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={(e) => onChange(parseFloat(e.target.value))}
            className={`h-1.5 w-full cursor-pointer appearance-none rounded-lg bg-secondary ${
              isRib ? "accent-[#00E599]" : "accent-[#38bdf8]"
            }`}
          />
          <button
            onClick={() => onChange(defaultValue)}
            disabled={value === defaultValue}
            title="Reset nilai ke default"
            className={`flex size-7 shrink-0 items-center justify-center rounded-md border text-xs transition-colors cursor-pointer ${
              value === defaultValue
                ? "opacity-30 cursor-not-allowed border-border"
                : "border-border hover:bg-muted text-foreground"
            }`}
          >
            <RotateCcw className="size-3" />
          </button>
        </div>
      </div>
    </div>
  );
}

// =========================================================================
// DIAGRAM ILUSTRASI SKEMATIK VISUAL (SVG GRAFIS)
// =========================================================================

/** 1. Diagram Tip Dev: Menunjukkan garis lurus vs ujung sirip yang menukik */
function TipDiagram({ value, isDark }: { value: number; isDark: boolean }) {
  return (
    <svg viewBox="0 0 400 90" className="w-full h-20 select-none">
      {/* Background Grid */}
      <line x1="20" y1="20" x2="380" y2="20" stroke={isDark ? "#334155" : "#e2e8f0"} strokeDasharray="3 3" />
      <line x1="20" y1="50" x2="380" y2="50" stroke={isDark ? "#334155" : "#e2e8f0"} strokeDasharray="3 3" />
      <line x1="20" y1="75" x2="380" y2="75" stroke={isDark ? "#334155" : "#e2e8f0"} strokeDasharray="3 3" />

      {/* Baseline Lurus Ideal (Cyan Dashed) */}
      <line x1="40" y1="35" x2="360" y2="35" stroke="#38bdf8" strokeWidth="1.5" strokeDasharray="5 3" />
      <text x="45" y="28" fill="#38bdf8" fontSize="9" fontWeight="bold">
        Baseline Lurus Acuan
      </text>

      {/* Kontur Fisik Sirip (Garis Merah): Badan lurus, ujung kanan menukik ke bawah */}
      <path
        d="M 50 35 L 300 35 C 320 35 340 55 355 65"
        fill="none"
        stroke="#ef4444"
        strokeWidth="2.5"
      />
      <circle cx="50" cy="35" r="3" fill="#ef4444" />
      <circle cx="355" cy="65" r="3.5" fill="#f59e0b" />

      {/* Panah Pengukuran Selisih Ujung (Delta Y) */}
      <line x1="365" y1="35" x2="365" y2="65" stroke="#f59e0b" strokeWidth="1.5" />
      <polygon points="365,33 362,38 368,38" fill="#f59e0b" />
      <polygon points="365,67 362,62 368,62" fill="#f59e0b" />

      <text x="280" y="78" fill="#f59e0b" fontSize="9" fontWeight="bold">
        Dev Tip: {value} px
      </text>
    </svg>
  );
}

/** 2. Diagram Body Dev: Menunjukkan koridor toleransi badan sirip */
function BodyDevDiagram({ value, isDark }: { value: number; isDark: boolean }) {
  return (
    <svg viewBox="0 0 400 90" className="w-full h-20 select-none">
      {/* Koridor Hijau Toleransi */}
      <rect x="50" y="25" width="300" height="30" fill={isDark ? "#10b98115" : "#10b98125"} rx="4" />
      <line x1="50" y1="25" x2="350" y2="25" stroke="#10b981" strokeWidth="1" strokeDasharray="3 3" />
      <line x1="50" y1="55" x2="350" y2="55" stroke="#10b981" strokeWidth="1" strokeDasharray="3 3" />

      {/* Baseline Tengah */}
      <line x1="40" y1="40" x2="360" y2="40" stroke="#38bdf8" strokeWidth="1.5" strokeDasharray="5 3" />

      {/* Kontur Riil Sirip (Garis Merah bergelombang tipis) */}
      <path
        d="M 50 40 Q 120 30 180 44 T 270 36 T 350 40"
        fill="none"
        stroke="#ef4444"
        strokeWidth="2.5"
      />

      <text x="55" y="20" fill="#10b981" fontSize="9" fontWeight="bold">
        Koridor Aman (Batas Simpangan: ±{value} px)
      </text>
      <text x="240" y="80" fill="#ef4444" fontSize="9" fontWeight="bold">
        Garis Puncak Riil
      </text>
    </svg>
  );
}

/** 3. Diagram Flatness PV: Selisih Puncak Tertinggi vs Terendah */
function FlatnessPVDiagram({ value, isDark }: { value: number; isDark: boolean }) {
  return (
    <svg viewBox="0 0 400 90" className="w-full h-20 select-none">
      {/* Garis Batas Atas Peak */}
      <line x1="50" y1="25" x2="310" y2="25" stroke="#38bdf8" strokeWidth="1.5" strokeDasharray="4 2" />
      <text x="55" y="20" fill="#38bdf8" fontSize="9" fontWeight="bold">
        Puncak Tertinggi (Peak)
      </text>

      {/* Garis Batas Bawah Valley */}
      <line x1="50" y1="65" x2="310" y2="65" stroke="#38bdf8" strokeWidth="1.5" strokeDasharray="4 2" />
      <text x="55" y="78" fill="#38bdf8" fontSize="9" fontWeight="bold">
        Lembah Terendah (Valley)
      </text>

      {/* Profil Fisik Sirip Miring/Melengkung */}
      <path d="M 60 55 C 130 20 220 70 300 45" fill="none" stroke="#ef4444" strokeWidth="2.5" />

      {/* Panah Vertikal PV Total */}
      <line x1="330" y1="25" x2="330" y2="65" stroke="#f59e0b" strokeWidth="2" />
      <polygon points="330,22 326,28 334,28" fill="#f59e0b" />
      <polygon points="330,68 326,62 334,62" fill="#f59e0b" />

      <text x="340" y="48" fill="#f59e0b" fontSize="10" fontWeight="bold">
        PV: {value} px
      </text>
    </svg>
  );
}

/** 4. Diagram Sagitta: Lengkungan Busur Panah */
function SagittaDiagram({ value, isDark }: { value: number; isDark: boolean }) {
  return (
    <svg viewBox="0 0 400 90" className="w-full h-20 select-none">
      {/* Garis Tali Busur Lurus (Chord) */}
      <line x1="60" y1="55" x2="340" y2="55" stroke="#38bdf8" strokeWidth="1.5" strokeDasharray="4 3" />
      <circle cx="60" cy="55" r="3" fill="#38bdf8" />
      <circle cx="340" cy="55" r="3" fill="#38bdf8" />

      {/* Bilah Sirip Melengkung Parabola */}
      <path d="M 60 55 Q 200 20 340 55" fill="none" stroke="#ef4444" strokeWidth="2.5" />

      {/* Panah Ketinggian Busur Sagitta di Tengah */}
      <line x1="200" y1="55" x2="200" y2="28" stroke="#f59e0b" strokeWidth="1.5" />
      <polygon points="200,25 197,31 203,31" fill="#f59e0b" />
      <polygon points="200,57 197,51 203,51" fill="#f59e0b" />

      <text x="210" y="45" fill="#f59e0b" fontSize="9" fontWeight="bold">
        Sagitta (Tinggi Lengkung): {value} px
      </text>
      <text x="65" y="72" fill="#38bdf8" fontSize="9">
        Garis Lurus Ujung ke Ujung
      </text>
    </svg>
  );
}

/** 5. Diagram AI YOLO Confidence */
function ConfidenceDiagram({ value, isDark }: { value: number; isDark: boolean }) {
  const pct = Math.round(value * 100);
  return (
    <svg viewBox="0 0 400 90" className="w-full h-20 select-none">
      {/* Kotak Bounding Box AI */}
      <rect x="50" y="20" width="300" height="50" rx="6" fill={isDark ? "#0284c715" : "#0284c710"} stroke="#38bdf8" strokeWidth="1.5" strokeDasharray="4 2" />
      <rect x="50" y="20" width="130" height="16" rx="4" fill="#0284c7" />
      <text x="56" y="32" fill="#ffffff" fontSize="9" fontWeight="bold">
        YOLO: RIB CONF {pct}%
      </text>
      <line x1="70" y1="45" x2="330" y2="45" stroke="#ef4444" strokeWidth="3" />
      <text x="180" y="62" fill={isDark ? "#94a3b8" : "#64748b"} fontSize="9">
        {value >= 0.5 ? "✓ Valid Sirip Terverifikasi" : "! Peringatan: Sensitivitas Rendah"}
      </text>
    </svg>
  );
}

/** 6. Diagram Slat Flatness */
function SlatFlatnessDiagram({ value, isDark }: { value: number; isDark: boolean }) {
  return (
    <svg viewBox="0 0 400 90" className="w-full h-20 select-none">
      {/* Permukaan Plat Slat */}
      <rect x="50" y="25" width="300" height="40" fill={isDark ? "#1e293b" : "#f1f5f9"} stroke="#64748b" strokeWidth="1" rx="4" />
      {/* Garis Sambungan Celah Nat */}
      <path d="M 60 45 C 140 40 220 52 340 45" fill="none" stroke="#10b981" strokeWidth="2.5" />
      {/* Indikator Toleransi PV */}
      <line x1="365" y1="40" x2="365" y2="52" stroke="#f59e0b" strokeWidth="1.5" />
      <text x="220" y="78" fill="#10b981" fontSize="9" fontWeight="bold">
        Celah Plat (Toleransi PV: {value} px)
      </text>
    </svg>
  );
}

/** 7. Diagram Slat Sagitta */
function SlatSagittaDiagram({ value, isDark }: { value: number; isDark: boolean }) {
  return (
    <svg viewBox="0 0 400 90" className="w-full h-20 select-none">
      <line x1="60" y1="50" x2="340" y2="50" stroke="#38bdf8" strokeWidth="1" strokeDasharray="3 3" />
      {/* Plat Melengkung */}
      <path d="M 60 50 Q 200 30 340 50" fill="none" stroke="#10b981" strokeWidth="2.5" />
      <line x1="200" y1="50" x2="200" y2="35" stroke="#f59e0b" strokeWidth="1.5" />
      <text x="210" y="44" fill="#f59e0b" fontSize="9" fontWeight="bold">
        Lengkung Plat: {value} px
      </text>
    </svg>
  );
}

/** 8. Diagram Void Gap Celah Sambungan */
function VoidGapDiagram({ value, isDark }: { value: number; isDark: boolean }) {
  return (
    <svg viewBox="0 0 400 90" className="w-full h-20 select-none">
      {/* Dua Plat Berdampingan */}
      <rect x="50" y="30" width="130" height="30" fill={isDark ? "#334155" : "#cbd5e1"} rx="3" />
      <rect x="220" y="30" width="130" height="30" fill={isDark ? "#334155" : "#cbd5e1"} rx="3" />
      {/* Area Celah di Tengah */}
      <rect x="180" y="25" width="40" height="40" fill={isDark ? "#f59e0b15" : "#f59e0b20"} stroke="#f59e0b" strokeWidth="1" strokeDasharray="3 2" />
      <text x="140" y="80" fill="#f59e0b" fontSize="9" fontWeight="bold">
        Batas Celah Renggang: {value} px
      </text>
    </svg>
  );
}
