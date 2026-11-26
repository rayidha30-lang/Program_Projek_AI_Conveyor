import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQueryState, parseAsString, parseAsStringLiteral } from "nuqs";
import {
  AlertTriangle,
  Ban,
  BarChart3,
  Calendar,
  Camera,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Clock3,
  Cpu,
  Download,
  ExternalLink,
  Eye,
  Folder,
  FolderOpen,
  Gauge,
  History,
  Layers,
  LayoutDashboard,
  LogOut,
  Maximize2,
  Menu,
  Minimize2,
  Minus,
  Moon,
  Plus,
  RotateCcw,
  Search,
  ShieldAlert,
  ShieldCheck,
  Sun,
  Wrench,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  fetchHealth,
  fetchStats,
  fetchInspections,
  fetchEvidence,
  fetchInspectionLogs,
  submitInspectionAction,
  fetchAnalytics,
  getImageUrl,
  extractRecordDate,
  extractEvidenceDate,
  formatPxAndMm,
  MONTH_NAMES_ID,
  MONTH_SHORT_ID,
  type RecordRow,
  type EvidenceItem,
  type MaintenanceLogItem,
  type AbnormalComponentSummary,
  type AnalyticsData,
  type Severity,
} from "@/lib/api";
import TuningSettings from "@/components/TuningSettings";
import {
  useLiveCameraStream,
  type CameraId,
  type CameraStreamState,
} from "@/hooks/useLiveCameraStream";

type View = "dashboard" | "live" | "abnormality" | "evidence" | "analytics" | "settings";

const abnormalComponentCounts: AbnormalComponentSummary[] = [
  {
    id: "plat-slat",
    label: "Total Abnormal Plat/Slat",
    shortLabel: "Plat / Slat",
    camera: "CAM-01",
    count: 5,
    dotColor: "bg-danger",
  },
  {
    id: "rib-sirip",
    label: "Total Abnormal Rib/Sirip",
    shortLabel: "Rib / Sirip",
    camera: "CAM-01",
    count: 3,
    dotColor: "bg-warning",
  },
  {
    id: "bolt-kanan",
    label: "Total Abnormal Bolt Sisi Kanan",
    shortLabel: "Bolt Kanan (RH)",
    camera: "CAM-02",
    count: 4,
    dotColor: "bg-primary",
  },
  {
    id: "pin-kanan",
    label: "Total Abnormal Pin Sisi Kanan",
    shortLabel: "Pin Kanan (RH)",
    camera: "CAM-02",
    count: 2,
    dotColor: "bg-info",
  },
  {
    id: "bolt-kiri",
    label: "Total Abnormal Bolt Sisi Kiri",
    shortLabel: "Bolt Kiri (LH)",
    camera: "CAM-03",
    count: 3,
    dotColor: "bg-primary",
  },
  {
    id: "pin-kiri",
    label: "Total Abnormal Pin Sisi Kiri",
    shortLabel: "Pin Kiri (LH)",
    camera: "CAM-03",
    count: 1,
    dotColor: "bg-danger",
  },
];

const records: RecordRow[] = [
  {
    date: "2026-10-20",
    displayDate: "20 Okt 2026",
    time: "10:31:22",
    id: "ABN-20261020-00021",
    inspectionId: 0,
    camera: "CAM-01",
    object: "Rib Plate #42",
    type: "Rib Deformation",
    confidence: "94.8%",
    severity: "CRITICAL",
    status: "UNACKNOWLEDGED",
    statusLifecycle: "UNACKNOWLEDGED",
  },
  {
    date: "2026-10-20",
    displayDate: "20 Okt 2026",
    time: "10:28:44",
    id: "ABN-20261020-00020",
    inspectionId: 0,
    camera: "CAM-02",
    object: "Fastener M16",
    type: "Bolt Missing",
    confidence: "97.2%",
    severity: "HIGH",
    status: "ACK: Rudi S.",
    statusLifecycle: "ACKNOWLEDGED",
  },
  {
    date: "2026-10-19",
    displayDate: "19 Okt 2026",
    time: "14:21:05",
    id: "ABN-20261019-00019",
    inspectionId: 0,
    camera: "CAM-03",
    object: "Connecting Pin",
    type: "Pin Missing",
    confidence: "91.4%",
    severity: "MEDIUM",
    status: "RESOLVED",
    statusLifecycle: "RESOLVED",
  },
  {
    date: "2026-10-17",
    displayDate: "17 Okt 2026",
    time: "11:14:18",
    id: "ABN-20261017-00018",
    inspectionId: 0,
    camera: "CAM-01",
    object: "Slat C-09",
    type: "Plate Deformation",
    confidence: "88.0%",
    severity: "LOW",
    status: "RESOLVED",
    statusLifecycle: "RESOLVED",
  },
];

const nav = [
  { id: "dashboard" as const, label: "Dashboard", icon: LayoutDashboard },
  { id: "live" as const, label: "Live Camera", icon: Camera },
  { id: "abnormality" as const, label: "Abnormality", icon: AlertTriangle },
  { id: "evidence" as const, label: "Evidence Viewer", icon: Eye },
  { id: "analytics" as const, label: "Analytics", icon: BarChart3 },
  { id: "settings" as const, label: "Tuning Settings", icon: Wrench },
];

const titles: Record<View, [string, string]> = {
  dashboard: [
    "AI Vision Inspection Dashboard",
    "Real-time AI Vision Inspection & Traceability System",
  ],
  live: [
    "Live Camera Monitoring",
    "Synchronous optical stream dari 3 sudut kamera inspeksi conveyor",
  ],
  abnormality: [
    "Abnormality History & Traceability Ledger",
    "Pencatatan riwayat kelainan, filter multi-parameter, dan audit trail",
  ],
  evidence: [
    "Evidence Viewer",
    "Arsip bukti optik inspeksi kamera & penelusuran status perbaikan",
  ],
  analytics: [
    "Analytics & Maintenance Insights",
    "Pola cacat berulang (recurring abnormality) & performa inspeksi",
  ],
  settings: [
    "Inspection Tuning & Threshold Settings",
    "Parameter ambang batas deteksi real-time (Auto Hot-Reload tanpa restart)",
  ],
};

export default function Index() {
  // Type-safe URL State via nuqs
  const [viewQuery, setViewQuery] = useQueryState(
    "view",
    parseAsStringLiteral(["dashboard", "live", "abnormality", "evidence", "analytics", "settings"]).withDefault("dashboard")
  );
  const [traceId, setTraceId] = useQueryState("trace", parseAsString);
  const [camId, setCamId] = useQueryState("cam", parseAsString);

  // Jika parameter trace atau cam terdeteksi di URL, utamakan tampilan evidence
  const view: View = (traceId || camId) ? "evidence" : viewQuery;
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const setView = (newView: View) => {
    setViewQuery(newView);
    setMobileMenuOpen(false);
    if (newView !== "evidence") {
      setTraceId(null);
      setCamId(null);
    }
  };

  const [isCollapsed, setIsCollapsed] = useState(false);

  useEffect(() => {
    const onResize = () => {
      if (window.innerWidth >= 1024) setMobileMenuOpen(false);
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMobileMenuOpen(false);
    };
    window.addEventListener("resize", onResize);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("resize", onResize);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, []);
  const [apiStatus, setApiStatus] = useState<{ status: string; database: string }>({
    status: "checking",
    database: "checking",
  });
  const [statsData, setStatsData] = useState<AbnormalComponentSummary[]>(abnormalComponentCounts);
  const [recordsData, setRecordsData] = useState<RecordRow[]>(records);
  const [evidenceData, setEvidenceData] = useState<EvidenceItem[]>([]);
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null);
  const {
    connected: wsConnected,
    showOverlay,
    toggleOverlay,
    streams,
  } = useLiveCameraStream(view === "dashboard" || view === "live");
  const [maximizedCam, setMaximizedCam] = useState<CameraId | null>(null);
  const [minimizedCams, setMinimizedCams] = useState<Record<CameraId, boolean>>({
    "CAM-01": false,
    "CAM-02": false,
    "CAM-03": false,
  });

  const toggleMinimizeCam = useCallback((cam: CameraId) => {
    setMinimizedCams((prev) => ({ ...prev, [cam]: !prev[cam] }));
  }, []);

  useEffect(() => {
    if (!maximizedCam) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMaximizedCam(null);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [maximizedCam]);

  // Pengendali tema (Dark / Light) dengan persistensi localStorage
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("dashboard_theme");
      if (saved === "dark" || saved === "light") return saved;
      return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    }
    return "light";
  });

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "dark") {
      root.classList.add("dark");
    } else {
      root.classList.remove("dark");
    }
    localStorage.setItem("dashboard_theme", theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

  const handleSelectRecord = (record: RecordRow) => {
    setTraceId(record.id);
    if ((record as any).camera) {
      setCamId((record as any).camera);
    }
    setViewQuery("evidence");
  };

  // Polling data secara real-time dari backend SQL Server
  const loadData = async () => {
    try {
      const h = await fetchHealth();
      setApiStatus(h);

      if (h.status === "online") {
        const [s, recs, ev, an] = await Promise.all([
          fetchStats().catch(() => null),
          fetchInspections().catch(() => null),
          fetchEvidence().catch(() => null),
          fetchAnalytics().catch(() => null),
        ]);

        if (s?.components && s.components.length > 0) setStatsData(s.components);
        if (recs && recs.length > 0) setRecordsData(recs);
        if (ev && ev.length > 0) setEvidenceData(ev);
        else setEvidenceData(evidence);
        if (an) setAnalyticsData(an);
      } else {
        setEvidenceData(evidence);
      }
    } catch (e) {
      console.error("Gagal sinkronisasi data:", e);
      setEvidenceData((prev) => (prev.length === 0 ? evidence : prev));
    }
  };

  useEffect(() => {
    let intervalId: any;

    const startPolling = (ms: number) => {
      if (intervalId) clearInterval(intervalId);
      intervalId = setInterval(loadData, ms);
    };

    const handleVisibilityChange = () => {
      if (document.hidden) {
        // Saat tab di background / operator fokus di GUI OpenCV, perlambat polling agar tidak membebani CPU & SQLite
        startPolling(20000);
      } else {
        // Saat operator beralih kembali membuka dashboard, segera fetch data terbaru dan kembalikan polling aktif
        loadData();
        startPolling(4000);
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    loadData();
    startPolling(4000);

    return () => {
      if (intervalId) clearInterval(intervalId);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, []);

  return (
    <div className="min-h-screen overflow-x-hidden bg-background text-foreground lg:h-screen lg:overflow-hidden transition-colors duration-300">
      <Sidebar
        view={view}
        setView={setView}
        isCollapsed={isCollapsed}
        setIsCollapsed={setIsCollapsed}
        mobileOpen={mobileMenuOpen}
        onCloseMobile={() => setMobileMenuOpen(false)}
        theme={theme}
        onToggleTheme={toggleTheme}
      />
      <div
        className={`min-w-0 transition-all duration-300 ease-in-out flex min-h-screen flex-col lg:h-screen ${
          isCollapsed ? "lg:ml-24" : "lg:ml-72"
        }`}
      >
        <Header
          view={view}
          apiStatus={apiStatus}
          wsConnected={wsConnected}
          theme={theme}
          onToggleTheme={toggleTheme}
          onOpenMobileMenu={() => setMobileMenuOpen(true)}
          contextActions={
            view === "live" ? (
              <div className="flex items-center gap-1.5 flex-wrap">
                <button
                  type="button"
                  onClick={toggleOverlay}
                  className="cursor-pointer"
                  title="Toggle overlay bounding box & kontur deteksi"
                >
                  <StatusPill
                    text={showOverlay ? "Bounding Box: ON" : "Bounding Box: OFF"}
                    online={showOverlay}
                  />
                </button>
                <span className="hidden sm:inline-flex">
                  <StatusPill
                    text={wsConnected ? "WS Stream: Live" : "WS Stream: Standby"}
                    online={wsConnected}
                  />
                </span>
              </div>
            ) : view === "evidence" ? (
              <div className="flex items-center gap-1.5">
                {traceId && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setTraceId(null);
                      setCamId(null);
                      setViewQuery("abnormality");
                    }}
                    className="h-7 text-[10px] gap-1 font-semibold text-primary border-primary/30 hover:bg-primary/10 cursor-pointer"
                  >
                    <ChevronLeft className="size-3" />
                    Kembali ke Histori
                  </Button>
                )}
                <div className="hidden md:flex items-center gap-1.5">
                  <StatusPill text="3 Target Camera Directories" />
                  <StatusPill text="Sync Storage" />
                </div>
              </div>
            ) : undefined
          }
        />
        <main className="flex-1 overflow-y-auto px-4 py-5 sm:px-6 lg:px-8 lg:py-7">
          {view === "dashboard" && (
            <Dashboard
              onViewAll={() => setView("abnormality")}
              onSelectRecord={handleSelectRecord}
              statsList={statsData}
              recentRecords={recordsData}
              streams={streams}
              wsConnected={wsConnected}
              minimizedCams={minimizedCams}
              onToggleMinimize={toggleMinimizeCam}
              onMaximize={setMaximizedCam}
            />
          )}
          {view === "live" && (
            <LiveCamera
              streams={streams}
              wsConnected={wsConnected}
              minimizedCams={minimizedCams}
              onToggleMinimize={toggleMinimizeCam}
              onMaximize={setMaximizedCam}
            />
          )}
          {view === "abnormality" && (
            <Abnormality
              allRecords={recordsData}
              onSelectRecord={handleSelectRecord}
            />
          )}
          {view === "evidence" && (
            <Evidence
              evidenceList={evidenceData}
              onRefresh={loadData}
              selectedTraceId={traceId}
              selectedCamera={camId}
              onSelectTraceId={setTraceId}
              onSelectCamera={setCamId}
              onBackToHistory={() => {
                setTraceId(null);
                setCamId(null);
                setViewQuery("abnormality");
              }}
            />
          )}
          {view === "analytics" && <Analytics data={analyticsData} />}
          {view === "settings" && <TuningSettings theme={theme} />}
        </main>
      </div>
      {maximizedCam && (
        <CameraSpotlightModal
          camera={maximizedCam}
          streamState={streams[maximizedCam]}
          wsConnected={wsConnected}
          showOverlay={showOverlay}
          onToggleOverlay={toggleOverlay}
          onMinimizeToGrid={() => {
            const cam = maximizedCam;
            setMaximizedCam(null);
            setMinimizedCams((prev) => ({ ...prev, [cam]: false }));
          }}
          onClose={() => setMaximizedCam(null)}
        />
      )}
    </div>
  );
}

function Sidebar({
  view,
  setView,
  isCollapsed,
  setIsCollapsed,
  mobileOpen,
  onCloseMobile,
  theme,
  onToggleTheme,
}: {
  view: View;
  setView: (view: View) => void;
  isCollapsed: boolean;
  setIsCollapsed: React.Dispatch<React.SetStateAction<boolean>>;
  mobileOpen: boolean;
  onCloseMobile: () => void;
  theme: "light" | "dark";
  onToggleTheme?: () => void;
}) {
  const navigate = useNavigate();
  const userName = localStorage.getItem("userName") || "Sarah Jenkins";
  const userInitials = userName
    .split(" ")
    .map((n) => n[0])
    .join("")
    .substring(0, 2)
    .toUpperCase();

  const handleLogout = () => {
    localStorage.removeItem("isAuthenticated");
    navigate("/login");
  };

  const isDark = theme === "dark";
  // Di drawer mobile/tablet (< lg), tampilkan selalu expanded agar label mudah dibaca
  const compactMode = isCollapsed && !mobileOpen;

  return (
    <>
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 backdrop-blur-xs lg:hidden"
          onClick={onCloseMobile}
          aria-hidden="true"
        />
      )}
      <aside
        className={`fixed inset-y-3 left-3 z-50 lg:z-30 flex flex-col justify-between overflow-y-auto rounded-3xl transition-all duration-300 ease-in-out ${
          mobileOpen ? "translate-x-0" : "-translate-x-[calc(100%+1.5rem)] lg:translate-x-0"
        } ${
          isDark
            ? "sidebar-gradient-emerald border border-emerald-400/40 text-white shadow-2xl shadow-emerald-950/60"
            : "bg-linear-to-b from-[#7645f7] via-[#6a37eb] to-[#5a24e2] border border-white/20 text-white shadow-2xl shadow-purple-900/30"
        } ${compactMode ? "w-20 py-5 px-2 items-center" : "w-64 py-5 px-4"}`}
      >
        <div>
          {compactMode ? (
            <div className="flex flex-col items-center gap-3">
              {/* Logo Cloud / Vision icon */}
              <button
                onClick={() => setIsCollapsed(false)}
                className={`flex size-11 items-center justify-center rounded-2xl shadow-sm transition-all cursor-pointer ${
                  isDark
                    ? "bg-black/25 text-slate-950 border border-black/15 hover:bg-black/35"
                    : "bg-white/20 text-white hover:bg-white/30"
                }`}
                title="Perbesar Side Bar"
              >
                <Cpu className={`size-6 ${isDark ? "text-slate-950" : "text-white"}`} />
              </button>
              <button
                onClick={() => setIsCollapsed(false)}
                className={`flex size-7 items-center justify-center rounded-lg transition-colors cursor-pointer ${
                  isDark
                    ? "bg-black/15 text-slate-950 hover:bg-black/25"
                    : "bg-white/10 text-white/80 hover:bg-white/25 hover:text-white"
                }`}
                title="Perbesar Side Bar"
              >
                <ChevronRight className="size-4" />
              </button>
            </div>
          ) : (
            <div className="mb-6 flex items-center justify-between gap-2">
              <div className="flex items-center gap-3 min-w-0">
                <div
                  className={`flex size-10 shrink-0 items-center justify-center rounded-2xl shadow-sm ${
                    isDark
                      ? "bg-black/20 text-slate-950 border border-black/15"
                      : "bg-white/20 text-white"
                  }`}
                >
                  <Cpu className={`size-5 ${isDark ? "text-slate-950" : "text-white"}`} />
                </div>
                <div className="overflow-hidden">
                  <p
                    className={`text-xs leading-tight truncate ${
                      isDark ? "font-black text-slate-950" : "font-extrabold text-white"
                    }`}
                  >
                    Die Cast Machine #2
                  </p>
                  <p
                    className={`mt-0.5 text-[8.5px] tracking-wider truncate ${
                      isDark
                        ? "font-black text-emerald-950/90"
                        : "font-bold text-purple-200"
                    }`}
                  >
                    SLAT RIB VISION SYSTEM
                  </p>
                </div>
              </div>
              <button
                onClick={() => setIsCollapsed(true)}
                className={`hidden lg:flex size-8 shrink-0 items-center justify-center rounded-lg transition-colors cursor-pointer ${
                  isDark
                    ? "bg-black/15 text-slate-950 hover:bg-black/25"
                    : "bg-white/10 text-white/80 hover:bg-white/20 hover:text-white"
                }`}
                title="Perkecil Side Bar"
              >
                <ChevronLeft className="size-4" />
              </button>
              <button
                type="button"
                onClick={onCloseMobile}
                className={`flex lg:hidden size-8 shrink-0 items-center justify-center rounded-lg transition-colors cursor-pointer ${
                  isDark
                    ? "bg-black/15 text-slate-950 hover:bg-black/25"
                    : "bg-white/10 text-white/80 hover:bg-white/20 hover:text-white"
                }`}
                title="Tutup Menu"
              >
                <X className="size-4" />
              </button>
            </div>
          )}

          {/* Navigation Items */}
          <nav
            className={`mt-4 flex ${
              compactMode
                ? "flex-col items-center gap-2.5"
                : "flex-col space-y-1.5"
            }`}
            aria-label="Main navigation"
          >
            {nav.map((item, idx) => {
              const Icon = item.icon;
              const active = view === item.id;
              const isUpper = idx < 2;

              let activeClass = "";
              let inactiveClass = "";

              if (isDark) {
                activeClass = isUpper
                  ? "bg-black/30 text-slate-950 font-black shadow-md ring-1 ring-black/20"
                  : "bg-[#00E599] text-slate-950 font-black shadow-lg shadow-emerald-500/35 ring-1 ring-emerald-300";
                inactiveClass = isUpper
                  ? "text-slate-950 font-bold hover:bg-black/15 hover:text-black"
                  : "text-slate-100 font-semibold hover:bg-white/15 hover:text-white";
              } else {
                activeClass = compactMode
                  ? "bg-black/25 text-white font-bold shadow-inner ring-1 ring-white/30"
                  : "bg-black/20 text-white font-bold shadow-inner ring-1 ring-white/30";
                inactiveClass = "text-white/80 font-bold hover:bg-white/15 hover:text-white";
              }

              return (
                <button
                  key={item.id}
                  onClick={() => setView(item.id)}
                  title={item.label}
                  className={`group relative flex items-center transition-all duration-200 cursor-pointer ${
                    compactMode
                      ? `size-11 justify-center rounded-2xl ${
                          active ? activeClass : inactiveClass
                        }`
                      : `h-11 w-full gap-3 rounded-xl px-3.5 text-xs ${
                          active ? activeClass : inactiveClass
                        }`
                  }`}
                >
                  <Icon
                    className={`${compactMode ? "size-5" : "size-4.5"} shrink-0`}
                  />
                  {!compactMode && <span className="truncate">{item.label}</span>}
                  {compactMode && (
                    <span className="pointer-events-none absolute left-full ml-3 z-50 whitespace-nowrap rounded-lg bg-gray-900/95 backdrop-blur-sm px-2.5 py-1 text-[11px] font-semibold text-white opacity-0 shadow-xl transition-opacity group-hover:opacity-100">
                      {item.label}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Bottom Footer Section */}
        {compactMode ? (
        <div className="flex flex-col items-center gap-3">
          {/* Quick Theme Toggle in Collapsed */}
          {onToggleTheme && (
            <button
              onClick={onToggleTheme}
              className={`flex size-10 items-center justify-center rounded-xl transition-all cursor-pointer ${
                isDark
                  ? "bg-white/10 text-amber-400 hover:bg-white/20 hover:text-amber-300"
                  : "bg-white/15 text-white hover:bg-white/25"
              }`}
              title={isDark ? "Ganti ke Mode Terang (Light Mode)" : "Ganti ke Mode Gelap (Dark Mode)"}
            >
              {isDark ? <Sun className="size-4.5" /> : <Moon className="size-4.5" />}
            </button>
          )}

          {/* User Avatar */}
          <div
            className={`group relative flex size-10 items-center justify-center overflow-hidden rounded-xl border-2 shadow-md cursor-pointer ${
              isDark
                ? "border-emerald-400/50 bg-black/30"
                : "border-white/50 bg-white/20"
            }`}
            title={`${userName} (QA Lead / PIC)`}
          >
            <img
              src="https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=120&auto=format&fit=crop&q=80"
              alt={userName}
              className="size-full object-cover"
              onError={(e) => {
                (e.target as HTMLElement).style.display = "none";
              }}
            />
            <span className="absolute text-[11px] font-extrabold text-white">
              {userInitials}
            </span>
          </div>

          {/* Logout Icon */}
          <button
            onClick={handleLogout}
            className={`flex size-10 items-center justify-center rounded-xl transition-colors cursor-pointer ${
              isDark
                ? "text-slate-300 hover:bg-white/15 hover:text-white"
                : "text-white/80 hover:bg-white/20 hover:text-white"
            }`}
            title="Logout System"
          >
            <LogOut className="size-4.5" />
          </button>
        </div>
      ) : (
        <div className="space-y-3 pt-4">
          <div
            className={`rounded-xl border p-3 backdrop-blur-md shadow-sm transition-colors ${
              isDark
                ? "border-white/15 bg-black/40 text-white"
                : "border-white/20 bg-white/10 text-white"
            }`}
          >
            <div className="mb-1.5 flex items-center justify-between text-[10px]">
              <span
                className={
                  isDark ? "text-emerald-400 font-semibold" : "text-purple-200"
                }
              >
                System Status
              </span>
              <span className="flex items-center gap-1 font-bold text-white">
                <span className="size-1.5 rounded-full bg-emerald-400 animate-pulse" />{" "}
                ONLINE
              </span>
            </div>
            <div
              className={`flex justify-between font-mono text-[9px] ${
                isDark ? "text-slate-300" : "text-purple-200"
              }`}
            >
              <span>Uptime: 99.8%</span>
              <span>SSMS Connected</span>
            </div>
          </div>

          <div className="flex items-center justify-between pt-1">
            <div className="flex items-center gap-2.5 overflow-hidden">
              <div
                className={`size-9 shrink-0 overflow-hidden rounded-xl border-2 shadow-sm relative flex items-center justify-center text-xs font-bold ${
                  isDark
                    ? "border-emerald-400/50 bg-black/30"
                    : "border-white/50 bg-white/20"
                }`}
              >
                <img
                  src="https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=120&auto=format&fit=crop&q=80"
                  alt={userName}
                  className="size-full object-cover"
                  onError={(e) => {
                    (e.target as HTMLElement).style.display = "none";
                  }}
                />
                <span className="absolute text-[10px] font-extrabold text-white">
                  {userInitials}
                </span>
              </div>
              <div className="truncate">
                <p className="text-[11px] font-extrabold text-white leading-tight truncate">
                  {userName}
                </p>
                <p
                  className={`text-[9px] font-medium truncate ${
                    isDark ? "text-emerald-300/80" : "text-purple-200"
                  }`}
                >
                  QA Lead / PIC
                </p>
              </div>
            </div>
            <div className="flex items-center gap-1">
              {onToggleTheme && (
                <button
                  type="button"
                  onClick={onToggleTheme}
                  className={`flex size-8 shrink-0 items-center justify-center rounded-lg transition-colors cursor-pointer ${
                    isDark
                      ? "text-amber-400 hover:bg-white/15 hover:text-amber-300"
                      : "text-white/80 hover:bg-white/20 hover:text-white"
                  }`}
                  title={isDark ? "Ganti ke Mode Terang" : "Ganti ke Mode Gelap"}
                >
                  {isDark ? <Sun className="size-4" /> : <Moon className="size-4" />}
                </button>
              )}
              <button
                onClick={handleLogout}
                className={`flex size-8 shrink-0 items-center justify-center rounded-lg transition-colors cursor-pointer ${
                  isDark
                    ? "text-slate-300 hover:bg-white/15 hover:text-white"
                    : "text-white/80 hover:bg-white/20 hover:text-white"
                }`}
                title="Logout System"
              >
                <LogOut className="size-4" />
              </button>
            </div>
          </div>
        </div>
      )}
      </aside>
    </>
  );
}

function Header({
  view,
  apiStatus,
  wsConnected,
  theme,
  onToggleTheme,
  onOpenMobileMenu,
  contextActions,
}: {
  view: View;
  apiStatus: { status: string; database: string };
  wsConnected: boolean;
  theme: "light" | "dark";
  onToggleTheme: () => void;
  onOpenMobileMenu: () => void;
  contextActions?: React.ReactNode;
}) {
  const [timeStr, setTimeStr] = useState("");

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTimeStr(
        now.toLocaleDateString("id-ID", {
          day: "2-digit",
          month: "short",
          year: "numeric",
        }) +
          " | " +
          now.toLocaleTimeString("id-ID", {
            hour: "2-digit",
            minute: "2-digit",
          }) +
          " WIB"
      );
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  const isOnline = apiStatus.status === "online";
  const isDb = apiStatus.database === "connected";
  const isDark = theme === "dark";

  return (
    <header className="flex flex-wrap items-center justify-between gap-3 px-4 pt-4 pb-2 sm:px-6 sm:pt-5 lg:px-8 bg-transparent">
      <div className="flex items-center gap-2.5 min-w-0">
        <button
          type="button"
          onClick={onOpenMobileMenu}
          aria-label="Buka menu navigasi"
          title="Menu Navigasi"
          className="inline-flex lg:hidden size-9 shrink-0 items-center justify-center rounded-xl border border-border bg-card text-foreground shadow-xs transition hover:bg-muted cursor-pointer"
        >
          <Menu className="size-4.5" />
        </button>
        <div className="min-w-0">
          <h1 className="truncate text-base font-extrabold sm:text-lg tracking-tight">
            {titles[view][0]}
          </h1>
          <p className="hidden truncate text-[10px] text-muted-foreground sm:block">
            {titles[view][1]}
          </p>
        </div>
      </div>
      <div className="flex items-center gap-1.5 sm:gap-2 flex-wrap">
        {/* Contextual Badges / Actions per View */}
        {contextActions}

        {/* Global System Status Pills */}
        <div className="hidden items-center gap-1.5 sm:flex sm:flex-wrap">
          <StatusPill text={isOnline ? "API: Online" : "API: Offline"} online={isOnline} />
          <span className="hidden md:inline-flex">
            <StatusPill text={isDb ? "SSMS: Connected" : "SSMS: Disconnected"} online={isDb} />
          </span>
          <StatusPill
            text={wsConnected ? "3 Cams: Live WS" : "3 Cams: Standby"}
            online={wsConnected}
          />
          <span className="hidden xl:flex items-center gap-1.5 rounded-lg border border-border/60 bg-card px-2.5 py-1.5 font-mono text-[9px] font-semibold text-muted-foreground shadow-xs">
            <Clock3 className="size-3" />
            {timeStr || "15 Sep 2026 | 09:00 WIB"}
          </span>
        </div>

        {/* Theme Toggle Button */}
        <button
          type="button"
          onClick={onToggleTheme}
          className={`group flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-[11px] font-bold shadow-xs transition-all duration-300 cursor-pointer ${
            isDark
              ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 hover:border-emerald-400/50 shadow-emerald-950/20"
              : "border-border/80 bg-card text-foreground hover:bg-muted/70 hover:border-border"
          }`}
          title={isDark ? "Ganti ke Mode Terang (Light Mode)" : "Ganti ke Mode Gelap (Dark Mode)"}
        >
          {isDark ? (
            <>
              <Sun className="size-3.5 text-amber-400 transition-transform duration-300 group-hover:rotate-45" />
              <span className="font-bold text-[10px]">Light</span>
            </>
          ) : (
            <>
              <Moon className="size-3.5 text-indigo-500 transition-transform duration-300 group-hover:-rotate-12" />
              <span className="font-bold text-[10px]">Dark</span>
            </>
          )}
        </button>
      </div>
    </header>
  );
}

function StatusPill({ text, online = true }: { text: string; online?: boolean }) {
  return (
    <span
      className={`flex items-center gap-1 rounded-full border px-2 py-1 text-[9px] font-bold ${
        online
          ? "border-success/30 bg-success-soft text-success"
          : "border-destructive/30 bg-destructive/10 text-destructive"
      }`}
    >
      <span
        className={`size-1.5 rounded-full ${
          online ? "bg-success animate-pulse" : "bg-destructive"
        }`}
      />
      {text}
    </span>
  );
}

function Panel({
  children,
  className = "",
  onClick,
}: {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
}) {
  return (
    <section
      onClick={onClick}
      className={`rounded-xl border bg-card panel-shadow ${className}`}
    >
      {children}
    </section>
  );
}

function SectionTitle({
  title,
  subtitle,
  aside,
}: {
  title: string;
  subtitle: string;
  aside?: React.ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div>
        <h2 className="text-sm font-extrabold">{title}</h2>
        <p className="mt-0.5 text-[10px] text-muted-foreground">{subtitle}</p>
      </div>
      {aside}
    </div>
  );
}

function Dashboard({
  onViewAll,
  onSelectRecord,
  statsList,
  recentRecords,
  streams,
  wsConnected,
  minimizedCams,
  onToggleMinimize,
  onMaximize,
}: {
  onViewAll: () => void;
  onSelectRecord?: (record: RecordRow) => void;
  statsList: AbnormalComponentSummary[];
  recentRecords: RecordRow[];
  streams: Record<CameraId, CameraStreamState>;
  wsConnected: boolean;
  minimizedCams: Record<CameraId, boolean>;
  onToggleMinimize: (cam: CameraId) => void;
  onMaximize: (cam: CameraId) => void;
}) {
  const statsMeta: Record<
    string,
    { note: string; Icon: any; tone: string; bgTone: string }
  > = {
    "plat-slat": {
      note: "Plate Deform / Crack",
      Icon: Layers,
      tone: "text-danger",
      bgTone: "bg-danger-soft text-danger",
    },
    "rib-sirip": {
      note: "Rib Bending / Profile",
      Icon: AlertTriangle,
      tone: "text-warning",
      bgTone: "bg-warning-soft text-warning",
    },
    "bolt-kanan": {
      note: "Loose / Missing Bolt",
      Icon: Wrench,
      tone: "text-info",
      bgTone: "bg-info-soft text-info",
    },
    "pin-kanan": {
      note: "Dislodged Link Pin",
      Icon: ShieldAlert,
      tone: "text-primary",
      bgTone: "bg-primary/10 text-primary",
    },
    "bolt-kiri": {
      note: "Loose / Untorqued",
      Icon: Wrench,
      tone: "text-primary",
      bgTone: "bg-primary/10 text-primary",
    },
    "pin-kiri": {
      note: "Critical Missing Pin",
      Icon: AlertTriangle,
      tone: "text-danger",
      bgTone: "bg-danger-soft text-danger",
    },
  };

  const displayStats = statsList.map((s) => {
    const meta = statsMeta[s.id] || {
      note: "Abnormal",
      Icon: AlertTriangle,
      tone: "text-danger",
      bgTone: "bg-danger-soft text-danger",
    };
    const activeVal = s.activeCount ?? s.count;
    const resolvedVal = s.resolvedCount ?? Math.max(0, s.count - activeVal);
    return {
      label: s.label,
      value: activeVal.toString(),
      totalValue: s.count,
      resolvedValue: resolvedVal,
      camera: s.camera,
      note: meta.note,
      Icon: meta.Icon,
      tone: meta.tone,
      bgTone: meta.bgTone,
      dotColor: s.dotColor,
    };
  });

  const activeRecentRecords = recentRecords.filter((r) => r.overallResult !== "INVALID");

  return (
    <div className="space-y-6">
      <div className="grid gap-3.5 grid-cols-1 sm:grid-cols-2 md:grid-cols-3 2xl:grid-cols-6">
        {displayStats.map((stat) => {
          const Icon = stat.Icon;
          return (
            <Panel
              key={stat.label}
              className="p-4 transition-all hover:border-primary/50 hover:shadow-sm cursor-pointer"
              onClick={onViewAll}
            >
              <div className="flex justify-between items-start gap-2">
                <div className="space-y-1">
                  <p className="text-[10px] font-bold text-muted-foreground leading-tight">
                    {stat.label}
                  </p>
                  <div className="flex items-baseline gap-1.5">
                    <p className="text-2xl font-extrabold text-foreground">{stat.value}</p>
                    <span className="text-[9px] font-semibold text-muted-foreground">aktif</span>
                  </div>
                  <p className="text-[8.5px] font-mono text-muted-foreground">
                    Total: {stat.totalValue} • {stat.resolvedValue} Resolved
                  </p>
                </div>
                <span
                  className={`flex size-8 shrink-0 items-center justify-center rounded-lg ${stat.bgTone}`}
                >
                  <Icon className="size-4" />
                </span>
              </div>
              <div className="mt-3 flex items-center justify-between border-t pt-2 text-[9px]">
                <span className="flex items-center gap-1 font-mono text-muted-foreground">
                  <span className={`size-1.5 rounded-full ${stat.dotColor}`} />
                  {stat.camera}
                </span>
                <span className={`font-semibold ${stat.tone}`}>{stat.note}</span>
              </div>
            </Panel>
          );
        })}
      </div>

      <Panel className="p-5">
        <SectionTitle
          title="Live Camera Vision Feeds & Detection Overlay"
          subtitle="Tampilan sinkron 3 sudut optik dengan AI bounding box real-time"
          aside={
            <span className="text-[9px] font-mono text-muted-foreground">
              {wsConnected
                ? "640×360 (16:9) • Binary WebSocket Live"
                : "Standby • Menunggu Stream (:8765)"}
            </span>
          }
        />
        <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-3 items-start">
          <CameraFeed
            camera="CAM-01"
            area="Plate / Rib"
            object="Conveyor Plate / Rib"
            objectType="RIB"
            conf="-"
            fps="--"
            streamState={streams["CAM-01"]}
            isMinimized={minimizedCams["CAM-01"]}
            onToggleMinimize={() => onToggleMinimize("CAM-01")}
            onMaximize={() => onMaximize("CAM-01")}
          />
          <CameraFeed
            camera="CAM-02"
            area="Bolt / Pin RH Side"
            object="Bolt / Pin RH"
            objectType="BOLT / PIN"
            conf="-"
            fps="--"
            streamState={streams["CAM-02"]}
            isMinimized={minimizedCams["CAM-02"]}
            onToggleMinimize={() => onToggleMinimize("CAM-02")}
            onMaximize={() => onMaximize("CAM-02")}
          />
          <CameraFeed
            camera="CAM-03"
            area="Bolt / Pin LH Side"
            object="Bolt / Pin LH"
            objectType="BOLT / PIN"
            conf="-"
            fps="--"
            streamState={streams["CAM-03"]}
            isMinimized={minimizedCams["CAM-03"]}
            onToggleMinimize={() => onToggleMinimize("CAM-03")}
            onMaximize={() => onMaximize("CAM-03")}
          />
        </div>
      </Panel>
      <Panel className="overflow-hidden">
        <div className="p-5">
          <SectionTitle
            title="Latest Abnormality History"
            subtitle="Log insiden anomali terdeteksi dari REST API vision stream"
            aside={
              <Button
                variant="ghost"
                size="sm"
                onClick={onViewAll}
                className="text-primary hover:text-primary hover:bg-primary/10"
              >
                Lihat Semua Histori <ChevronRight className="size-4" />
              </Button>
            }
          />
        </div>
        <RecordTable
          rows={activeRecentRecords.slice(0, 5)}
          compact
          onSelectRecord={onSelectRecord}
        />
      </Panel>
    </div>
  );
}

const CAMERA_CATALOG: Record<
  CameraId,
  {
    area: string;
    station: string;
    title: string;
    desc: string;
    fallbackObj: string;
  }
> = {
  "CAM-01": {
    area: "Plate / Rib",
    station: "Station: ST-01 / Overhead",
    title: "Plat / Rib Inspection Area",
    desc: "Mendeteksi kelurusan celah plat slat & deformasi profil rib conveyor.",
    fallbackObj: "Conveyor Plate / Rib",
  },
  "CAM-02": {
    area: "Bolt / Pin RH Side",
    station: "Station: ST-01 / Right Oblique",
    title: "Bolt / Pin Right Side Area",
    desc: "Inspeksi baut penahan samping kanan, pin, dan retakan bracket.",
    fallbackObj: "Bolt / Pin RH",
  },
  "CAM-03": {
    area: "Bolt / Pin LH Side",
    station: "Station: ST-01 / Left Oblique",
    title: "Bolt / Pin Left Side Area",
    desc: "Inspeksi baut penahan samping kiri, pin, dan retakan bracket.",
    fallbackObj: "Bolt / Pin LH",
  },
};

function CameraWindowButtons({
  camera,
  isMinimized,
  onToggleMinimize,
  onMaximize,
}: {
  camera: string;
  isMinimized: boolean;
  onToggleMinimize: () => void;
  onMaximize: () => void;
}) {
  return (
    <div className="flex items-center gap-1">
      <button
        type="button"
        onClick={onToggleMinimize}
        title={isMinimized ? `Tampilkan preview ${camera}` : `Minimize preview ${camera}`}
        className="inline-flex size-6 items-center justify-center rounded border border-border/70 bg-background/70 text-muted-foreground transition hover:bg-accent hover:text-foreground"
      >
        {isMinimized ? <Plus className="size-3.5" /> : <Minus className="size-3.5" />}
      </button>
      <button
        type="button"
        onClick={onMaximize}
        title={`Maximize ${camera} ke tampilan besar`}
        className="inline-flex size-6 items-center justify-center rounded border border-border/70 bg-background/70 text-muted-foreground transition hover:bg-accent hover:text-foreground"
      >
        <Maximize2 className="size-3" />
      </button>
    </div>
  );
}

const CameraFeed = React.memo(function CameraFeed({
  camera,
  area,
  object,
  objectType,
  conf,
  fps = "--",
  sev,
  alert = false,
  streamState,
  isMinimized = false,
  onToggleMinimize,
  onMaximize,
}: {
  camera: CameraId;
  area: string;
  object: string;
  objectType: string;
  conf: string;
  fps?: string;
  sev?: string;
  alert?: boolean;
  streamState?: CameraStreamState;
  isMinimized?: boolean;
  onToggleMinimize: () => void;
  onMaximize: () => void;
}) {
  const liveMeta = streamState?.meta;
  const frameUrl = streamState?.frameUrl;
  const hasLive = Boolean(frameUrl);
  const isAlarm = hasLive ? liveMeta?.status === "NG" : alert;
  const statusText = hasLive
    ? isAlarm
      ? "ALARM ACTIVE"
      : liveMeta?.status || "ONLINE"
    : "STANDBY";
  const displayFps = hasLive && liveMeta?.fps !== undefined ? liveMeta.fps.toFixed(1) : fps;
  const displayObject = hasLive && liveMeta?.object ? liveMeta.object : objectType;
  const displayConf =
    hasLive && liveMeta?.conf !== null && liveMeta?.conf !== undefined
      ? `${(liveMeta.conf * 100).toFixed(1)}%`
      : conf;
  const displaySev =
    hasLive && liveMeta?.severity && liveMeta.severity !== "NONE"
      ? liveMeta.severity
      : sev;

  return (
    <div
      className={`overflow-hidden rounded-lg border ${
        isAlarm ? "border-danger/50" : "border-border"
      }`}
    >
      <div className="flex items-center justify-between gap-2 bg-secondary px-3 py-2 text-[10px]">
        <span className="truncate font-bold">
          {camera}{" "}
          <span className="font-medium text-muted-foreground">({area})</span>
        </span>
        <div className="flex items-center gap-2 shrink-0">
          <span
            className={`font-bold ${
              !hasLive
                ? "text-muted-foreground"
                : isAlarm
                  ? "text-danger"
                  : "text-success"
            }`}
          >
            {statusText}
          </span>
          <CameraWindowButtons
            camera={camera}
            isMinimized={isMinimized}
            onToggleMinimize={onToggleMinimize}
            onMaximize={onMaximize}
          />
        </div>
      </div>
      {!isMinimized && (
        <div
          onDoubleClick={onMaximize}
          title="Double-click atau klik Maximize untuk memperbesar"
          className="camera-grid relative flex aspect-video items-center justify-center overflow-hidden bg-black/90 cursor-pointer"
        >
          {frameUrl ? (
            <img
              src={frameUrl}
              alt={`${camera} live stream`}
              className="size-full object-cover"
            />
          ) : (
            <>
              <div
                className={`absolute inset-x-0 top-0 h-px ${
                  isAlarm ? "bg-danger" : "bg-muted-foreground/40"
                } scan-line`}
              />
              <div className="flex flex-col items-center gap-1.5 text-center px-4">
                <Camera className="size-6 text-muted-foreground/70" />
                <span className="font-mono text-[10px] text-muted-foreground">
                  Menunggu stream {camera} ({object})
                </span>
              </div>
            </>
          )}
        </div>
      )}
      <div className="flex flex-wrap items-center justify-between gap-1.5 bg-card px-3 py-2 font-mono text-[9px] text-muted-foreground border-t">
        <div className="flex items-center gap-1.5 sm:gap-2">
          <span>
            Object: <strong className="text-foreground">{displayObject}</strong>
          </span>
          <span className="text-border">|</span>
          <span>
            Conf: <strong className="text-foreground">{displayConf}</strong>
          </span>
          <span className="text-border">|</span>
          <span className="flex items-center gap-1 font-bold text-foreground">
            <span
              className={`size-1.5 rounded-full ${
                hasLive ? "bg-success animate-pulse" : "bg-muted-foreground/50"
              }`}
            />
            FPS: <strong className="text-foreground">{displayFps}</strong>
          </span>
        </div>
        {displaySev && (
          <span className="rounded border border-danger/30 bg-danger-soft px-1.5 py-0.5 text-[8px] font-bold text-danger">
            Sev: {displaySev}
          </span>
        )}
      </div>
    </div>
  );
});

const LiveCamera = React.memo(function LiveCamera({
  streams,
  wsConnected,
  minimizedCams,
  onToggleMinimize,
  onMaximize,
}: {
  streams: Record<CameraId, CameraStreamState>;
  wsConnected: boolean;
  minimizedCams: Record<CameraId, boolean>;
  onToggleMinimize: (cam: CameraId) => void;
  onMaximize: (cam: CameraId) => void;
}) {
  const camIds: CameraId[] = ["CAM-01", "CAM-02", "CAM-03"];

  return (
    <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3 items-start">
      {camIds.map((cam) => {
        const c = CAMERA_CATALOG[cam];
        const st = streams[cam];
        const meta = st?.meta;
        const hasLive = Boolean(st?.frameUrl);
        const isAlert = hasLive && meta?.status === "NG";
        const statusLabel = hasLive ? (meta?.status || "ONLINE") : "STANDBY";
        const fpsLabel = hasLive && meta?.fps !== undefined ? meta.fps.toFixed(1) : "--";
        const objLabel = hasLive && meta?.object ? meta.object : c.fallbackObj;
        const isMinimized = Boolean(minimizedCams[cam]);

        return (
          <Panel key={cam} className="overflow-hidden">
            <div className="flex items-start justify-between gap-2 p-4 pb-3">
              <div>
                <p className="text-xs font-extrabold">
                  {cam}{" "}
                  <span
                    className={
                      !hasLive
                        ? "text-muted-foreground"
                        : isAlert
                          ? "text-danger"
                          : "text-success"
                    }
                  >
                    {statusLabel}
                  </span>
                </p>
                <p className="mt-1 text-[9px] text-muted-foreground">
                  {c.station}
                </p>
              </div>
              <CameraWindowButtons
                camera={cam}
                isMinimized={isMinimized}
                onToggleMinimize={() => onToggleMinimize(cam)}
                onMaximize={() => onMaximize(cam)}
              />
            </div>
            {!isMinimized && (
              <div
                onDoubleClick={() => onMaximize(cam)}
                title="Double-click atau klik Maximize untuk memperbesar"
                className="camera-grid relative flex aspect-video items-center justify-center overflow-hidden bg-black/90 cursor-pointer"
              >
                {st?.frameUrl ? (
                  <img
                    src={st.frameUrl}
                    alt={`${cam} live view`}
                    className="size-full object-cover"
                  />
                ) : (
                  <div className="flex flex-col items-center gap-1.5 text-center px-4">
                    <Camera className="size-7 text-muted-foreground/60" />
                    <span className="font-mono text-[10px] text-muted-foreground">
                      {wsConnected
                        ? `Menunggu frame ${cam}...`
                        : "Standby (jalankan runner dengan --ws-stream)"}
                    </span>
                  </div>
                )}
                <div
                  className={`absolute bottom-2.5 left-2.5 rounded px-2 py-0.5 font-mono text-[9px] font-bold ${
                    !hasLive
                      ? "bg-black/70 text-muted-foreground border border-white/10"
                      : isAlert
                        ? "bg-danger text-primary-foreground"
                        : "bg-success text-primary-foreground"
                  }`}
                >
                  {!hasLive
                    ? "STREAM STANDBY"
                    : isAlert
                      ? "CRITICAL DEFECT DETECTED"
                      : "STREAM VIEW • OK"}
                </div>
              </div>
            )}
            <div className="p-4">
              {!isMinimized && (
                <>
                  <h3 className="text-sm font-bold">{c.title}</h3>
                  <p className="mt-1 text-[10px] text-muted-foreground">{c.desc}</p>
                </>
              )}
              <div
                className={`${
                  isMinimized ? "" : "mt-4 border-t pt-3"
                } flex justify-between text-[9px] font-mono`}
              >
                <span>
                  FPS: <b>{fpsLabel}</b> • 360p WS Live
                </span>
                <b className={isAlert ? "text-danger" : hasLive ? "text-success" : "text-muted-foreground"}>
                  {objLabel}
                </b>
              </div>
            </div>
          </Panel>
        );
      })}
    </div>
  );
});

function CameraSpotlightModal({
  camera,
  streamState,
  wsConnected,
  showOverlay,
  onToggleOverlay,
  onMinimizeToGrid,
  onClose,
}: {
  camera: CameraId;
  streamState?: CameraStreamState;
  wsConnected: boolean;
  showOverlay: boolean;
  onToggleOverlay: () => void;
  onMinimizeToGrid: () => void;
  onClose: () => void;
}) {
  const info = CAMERA_CATALOG[camera];
  const meta = streamState?.meta;
  const frameUrl = streamState?.frameUrl;
  const hasLive = Boolean(frameUrl);
  const isAlarm = hasLive && meta?.status === "NG";
  const statusText = hasLive
    ? isAlarm
      ? "ALARM ACTIVE"
      : meta?.status || "ONLINE"
    : "STANDBY";
  const displayFps = hasLive && meta?.fps !== undefined ? meta.fps.toFixed(1) : "--";
  const displayObject = hasLive && meta?.object ? meta.object : info.fallbackObj;
  const displayConf =
    hasLive && meta?.conf !== null && meta?.conf !== undefined
      ? `${(meta.conf * 100).toFixed(1)}%`
      : "-";
  const displaySev =
    hasLive && meta?.severity && meta.severity !== "NONE" ? meta.severity : null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-3 sm:p-6 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className={`w-full max-w-5xl max-h-[95vh] flex flex-col overflow-y-auto rounded-xl border bg-card shadow-2xl ${
          isAlarm ? "border-danger/60" : "border-border"
        }`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex flex-wrap items-center justify-between gap-3 border-b bg-secondary px-4 py-3">
          <div className="flex items-center gap-2.5">
            <span className="rounded bg-primary/15 px-2 py-0.5 text-xs font-extrabold text-primary">
              {camera}
            </span>
            <div>
              <p className="text-xs font-bold">
                {info.title}{" "}
                <span className="font-normal text-muted-foreground">({info.area})</span>
              </p>
              <p className="text-[10px] text-muted-foreground">{info.station}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`rounded px-2 py-0.5 text-[10px] font-bold ${
                !hasLive
                  ? "bg-muted text-muted-foreground"
                  : isAlarm
                    ? "bg-danger-soft text-danger"
                    : "bg-success-soft text-success"
              }`}
            >
              {statusText}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={onToggleOverlay}
              className="h-7 px-2.5 text-[10px]"
            >
              <Eye className="size-3.5 mr-1" />
              {showOverlay ? "Bounding Box: ON" : "Bounding Box: OFF"}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={onMinimizeToGrid}
              title="Kembalikan ke tampilan grid 3 kamera"
              className="h-7 px-2.5 text-[10px]"
            >
              <Minimize2 className="size-3.5 mr-1" />
              Minimize
            </Button>
            <button
              type="button"
              onClick={onClose}
              title="Tutup (Esc)"
              className="inline-flex size-7 items-center justify-center rounded-md border border-border bg-background text-muted-foreground hover:bg-accent hover:text-foreground"
            >
              <X className="size-4" />
            </button>
          </div>
        </div>

        <div className="relative flex aspect-video w-full items-center justify-center bg-black">
          {frameUrl ? (
            <img
              src={frameUrl}
              alt={`${camera} maximized live view`}
              className="size-full object-contain"
            />
          ) : (
            <div className="flex flex-col items-center gap-2 text-center px-6">
              <Camera className="size-10 text-muted-foreground/60" />
              <span className="font-mono text-xs text-muted-foreground">
                {wsConnected
                  ? `Menunggu frame ${camera}...`
                  : "Standby — jalankan runner dengan WebSocket aktif (:8765)"}
              </span>
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 border-t bg-card px-4 py-2.5 font-mono text-[11px] text-muted-foreground">
          <div className="flex flex-wrap items-center gap-3">
            <span>
              Object: <strong className="text-foreground">{displayObject}</strong>
            </span>
            <span className="text-border">|</span>
            <span>
              Conf: <strong className="text-foreground">{displayConf}</strong>
            </span>
            <span className="text-border">|</span>
            <span className="flex items-center gap-1.5 font-bold text-foreground">
              <span
                className={`size-2 rounded-full ${
                  hasLive ? "bg-success animate-pulse" : "bg-muted-foreground/50"
                }`}
              />
              FPS: <strong>{displayFps}</strong>
            </span>
            {displaySev && (
              <span className="rounded border border-danger/30 bg-danger-soft px-1.5 py-0.5 text-[10px] font-bold text-danger">
                Sev: {displaySev}
              </span>
            )}
          </div>
          <span className="text-[10px]">640×360 Binary WS • Tekan ESC untuk keluar</span>
        </div>
      </div>
    </div>
  );
}

function getSlidePageNumbers(current: number, total: number): (number | "...")[] {
  if (total <= 5) {
    return Array.from({ length: total }, (_, i) => i + 1);
  }
  const pages: (number | "...")[] = [];
  if (current <= 3) {
    pages.push(1, 2, 3, 4, "...", total);
  } else if (current >= total - 2) {
    pages.push(1, "...", total - 3, total - 2, total - 1, total);
  } else {
    pages.push(1, "...", current - 1, current, current + 1, "...", total);
  }
  return pages;
}

type LifecycleBucket = "OPEN" | "RESOLVED" | "INVALID";

function getRecordBucket(r: {
  overallResult?: string;
  detectionResult?: string;
  statusLifecycle?: string;
}): LifecycleBucket {
  if (r.overallResult === "INVALID" || r.detectionResult === "INVALID") return "INVALID";
  if (r.statusLifecycle === "RESOLVED") return "RESOLVED";
  return "OPEN";
}

function Abnormality({
  allRecords,
  onSelectRecord,
}: {
  allRecords: RecordRow[];
  onSelectRecord?: (record: RecordRow) => void;
}) {
  const [viewTab, setViewTab] = useState<"OPEN" | "RESOLVED" | "INVALID" | "ALL">("OPEN");
  const [camera, setCamera] = useState("ALL");
  const [severity, setSeverity] = useState("ALL");
  const [status, setStatus] = useState("ALL");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedYear, setSelectedYear] = useState("ALL");
  const [selectedMonth, setSelectedMonth] = useState("ALL");
  const [selectedDate, setSelectedDate] = useState("");

  // Slide / Pagination state
  // Default 10 riwayat per slide agar pas satu layar dashboard tanpa perlu scroll vertikal
  const [pageSize, setPageSize] = useState<number>(10);
  const [currentSlide, setCurrentSlide] = useState<number>(1);

  const tabCounts = useMemo(() => {
    const counts = { OPEN: 0, RESOLVED: 0, INVALID: 0, ALL: allRecords.length };
    allRecords.forEach((r) => {
      counts[getRecordBucket(r)] += 1;
    });
    return counts;
  }, [allRecords]);

  const availableYears = useMemo(() => {
    const years = new Set<string>();
    allRecords.forEach((r) => {
      const d = extractRecordDate(r);
      if (d.year) years.add(d.year.toString());
    });
    const arr = Array.from(years).sort().reverse();
    return arr.length > 0 ? arr : ["2026", "2025"];
  }, [allRecords]);

  const monthOptions = [
    { value: "01", label: "01 - Januari" },
    { value: "02", label: "02 - Februari" },
    { value: "03", label: "03 - Maret" },
    { value: "04", label: "04 - April" },
    { value: "05", label: "05 - Mei" },
    { value: "06", label: "06 - Juni" },
    { value: "07", label: "07 - Juli" },
    { value: "08", label: "08 - Agustus" },
    { value: "09", label: "09 - September" },
    { value: "10", label: "10 - Oktober" },
    { value: "11", label: "11 - November" },
    { value: "12", label: "12 - Desember" },
  ];

  const filtered = useMemo(() => {
    return allRecords.filter((r) => {
      const bucket = getRecordBucket(r);
      if (viewTab !== "ALL" && bucket !== viewTab) return false;
      if (camera !== "ALL" && r.camera !== camera) return false;
      if (severity !== "ALL" && r.severity !== severity) return false;
      if (status !== "ALL") {
        if (status === "FALSE ALARM") {
          if (bucket !== "INVALID") return false;
        } else if (!r.status.startsWith(status) && r.statusLifecycle !== status) {
          return false;
        }
      }

      const dateInfo = extractRecordDate(r);

      if (selectedYear !== "ALL" && dateInfo.year.toString() !== selectedYear) {
        return false;
      }

      if (selectedMonth !== "ALL") {
        const mStr = String(dateInfo.month).padStart(2, "0");
        if (mStr !== selectedMonth) return false;
      }

      if (selectedDate && dateInfo.date !== selectedDate) {
        return false;
      }

      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase().trim();
        const matchId = r.id.toLowerCase().includes(q);
        const matchObj = r.object.toLowerCase().includes(q);
        const matchType = r.type.toLowerCase().includes(q);
        if (!matchId && !matchObj && !matchType) return false;
      }

      return true;
    });
  }, [allRecords, viewTab, camera, severity, status, selectedYear, selectedMonth, selectedDate, searchQuery]);

  // Hitung total slide berdasarkan jumlah record hasil filter & pageSize
  const totalSlides = Math.max(1, Math.ceil(filtered.length / pageSize));

  // Auto-reset ke slide 1 ketika kriteria filter atau ukuran slide berubah
  useEffect(() => {
    setCurrentSlide(1);
  }, [viewTab, camera, severity, status, selectedYear, selectedMonth, selectedDate, searchQuery, pageSize]);

  // Proteksi bounds slide jika data berkurang akibat polling
  useEffect(() => {
    if (currentSlide > totalSlides) {
      setCurrentSlide(totalSlides);
    }
  }, [totalSlides, currentSlide]);

  const startIndex = (currentSlide - 1) * pageSize;
  const endIndex = Math.min(startIndex + pageSize, filtered.length);

  // Data baris aktif untuk slide saat ini
  const paginatedRecords = useMemo(() => {
    return filtered.slice(startIndex, endIndex);
  }, [filtered, startIndex, endIndex]);

  const isFiltered =
    camera !== "ALL" ||
    severity !== "ALL" ||
    status !== "ALL" ||
    selectedYear !== "ALL" ||
    selectedMonth !== "ALL" ||
    selectedDate !== "" ||
    searchQuery !== "";

  const handleResetFilters = () => {
    setCamera("ALL");
    setSeverity("ALL");
    setStatus("ALL");
    setSelectedYear("ALL");
    setSelectedMonth("ALL");
    setSelectedDate("");
    setSearchQuery("");
  };

  function exportCsv() {
    const head =
      "Tanggal,Waktu,Abnormality ID,Kamera,Objek,Tipe Kelainan,AI Confidence,Severity,Status,Waktu Resolved,PIC,Catatan Maintenance,Timestamp Penuh\n";
    const csv =
      head +
      filtered
        .map((r) => {
          const d = extractRecordDate(r);
          return [
            `"${d.displayDate}"`,
            `"${r.time}"`,
            `"${r.id}"`,
            `"${r.camera}"`,
            `"${r.object.replace(/"/g, '""')}"`,
            `"${r.type.replace(/"/g, '""')}"`,
            `"${r.confidence}"`,
            `"${r.severity}"`,
            `"${r.status.replace(/"/g, '""')}"`,
            `"${(r.resolvedAtDisplay || "").replace(/"/g, '""')}"`,
            `"${(r.picName || "").replace(/"/g, '""')}"`,
            `"${(r.maintenanceNote || "").replace(/"/g, '""')}"`,
            `"${(r.fullTime || "").replace(/"/g, '""')}"`,
          ].join(",");
        })
        .join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8;" }));
    a.download = `abnormality-history-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  const quickTabs: { id: "OPEN" | "RESOLVED" | "INVALID" | "ALL"; label: string; count: number; tone: string }[] = [
    { id: "OPEN", label: "Perlu Tindakan (Open)", count: tabCounts.OPEN, tone: "text-warning" },
    { id: "RESOLVED", label: "Arsip Selesai (Resolved)", count: tabCounts.RESOLVED, tone: "text-success" },
    { id: "INVALID", label: "False Alarm / Invalid", count: tabCounts.INVALID, tone: "text-muted-foreground" },
    { id: "ALL", label: "Semua Riwayat", count: tabCounts.ALL, tone: "text-primary" },
  ];

  return (
    <Panel className="overflow-hidden">
      <div className="p-3.5 pb-2.5 space-y-3">
        {/* Quick View Tabs: Open vs Resolved vs False Alarm vs All */}
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border/60 pb-2.5">
          <div className="flex flex-wrap items-center gap-1.5">
            {quickTabs.map((tab) => {
              const active = viewTab === tab.id;
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => {
                    setViewTab(tab.id);
                    setStatus("ALL");
                  }}
                  className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[10px] font-bold transition-all cursor-pointer border ${
                    active
                      ? "border-primary/40 bg-primary/10 text-primary shadow-2xs"
                      : "border-border/60 bg-secondary/40 text-muted-foreground hover:bg-secondary hover:text-foreground"
                  }`}
                >
                  <span>{tab.label}</span>
                  <span
                    className={`rounded-full px-1.5 py-0.2 font-mono text-[9px] ${
                      active ? "bg-primary/20 text-primary" : `bg-background/80 ${tab.tone}`
                    }`}
                  >
                    {tab.count}
                  </span>
                </button>
              );
            })}
          </div>
          <span className="text-[9px] text-muted-foreground hidden sm:inline">
            Data tersimpan sebagai audit trail (retensi otomatis 7 hari)
          </span>
        </div>

        {/* Filter Bar Multi-Parameter & Aksi Cepat */}
        <div className="flex items-center justify-between gap-2 overflow-x-auto custom-scrollbar pb-1">
          <div className="flex items-center gap-1.5 shrink-0">
            {/* 1. Pencarian Kata Kunci */}
            <div className="relative inline-flex items-center w-36 sm:w-40 xl:w-44 shrink-0">
              <Search className="absolute left-2.5 pointer-events-none size-3 text-muted-foreground" />
              <input
                type="text"
                placeholder="Cari ID, objek, tipe..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="h-8 w-full rounded-lg border bg-secondary/70 py-1 pl-7 pr-2 text-[9.5px] placeholder:text-muted-foreground outline-none focus:ring-1 focus:ring-primary text-foreground transition-all hover:bg-secondary"
              />
            </div>

            {/* 2. Filter Tanggal Spesifik (Date Picker) */}
            <div className="relative inline-flex items-center shrink-0">
              <Calendar className="absolute left-2.5 pointer-events-none size-3 text-muted-foreground" />
              <input
                type="date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
                className="h-8 w-[118px] rounded-lg border bg-secondary/70 py-1 pl-7 pr-1 text-[9.5px] font-semibold outline-none focus:ring-1 focus:ring-primary text-foreground transition-all hover:bg-secondary cursor-pointer"
                title="Filter tanggal spesifik (YYYY-MM-DD)"
              />
            </div>

            {/* 3. Filter Tahun */}
            <Filter
              value={selectedYear}
              setValue={setSelectedYear}
              label="Semua Tahun"
              options={availableYears.map((y) => ({ value: y, label: `Tahun ${y}` }))}
            />

            {/* 4. Filter Bulan */}
            <Filter
              value={selectedMonth}
              setValue={setSelectedMonth}
              label="Semua Bulan"
              options={monthOptions}
            />

            {/* 5. Filter Kamera */}
            <Filter
              value={camera}
              setValue={setCamera}
              label="Semua Kamera"
              options={["CAM-01", "CAM-02", "CAM-03"]}
            />

            {/* 6. Filter Severity */}
            <Filter
              value={severity}
              setValue={setSeverity}
              label="Semua Severity"
              options={["CRITICAL", "HIGH", "MEDIUM", "LOW"]}
            />

            {/* 7. Filter Status */}
            <Filter
              value={status}
              setValue={setStatus}
              label="Semua Status"
              options={["UNACKNOWLEDGED", "ACK", "RESOLVED", "FALSE ALARM"]}
            />

            {/* 8. Tombol Reset Filter */}
            {isFiltered && (
              <Button
                size="sm"
                variant="ghost"
                onClick={handleResetFilters}
                className="h-8 text-[9.5px] text-muted-foreground hover:text-foreground gap-1 px-2 shrink-0 cursor-pointer"
                title="Reset semua filter ke kondisi awal"
              >
                <RotateCcw className="size-3" />
                Reset
              </Button>
            )}
          </div>

          {/* Sisi Kanan: Total Counter & Tombol Export CSV */}
          <div className="flex items-center gap-2 shrink-0 ml-2">
            <span className="text-[9.5px] font-mono text-muted-foreground hidden lg:inline whitespace-nowrap">
              Total: <b className="text-foreground">{filtered.length}</b> dari {allRecords.length}
            </span>
            <Button size="sm" onClick={exportCsv} className="h-8 text-[11px] gap-1.5 px-3 shrink-0 cursor-pointer whitespace-nowrap font-bold">
              <Download className="size-3.5" />
              Export CSV
            </Button>
          </div>
        </div>
      </div>

      <RecordTable rows={paginatedRecords} compact onSelectRecord={onSelectRecord} />

      {filtered.length === 0 ? (
        <div className="p-8 text-center space-y-2">
          <p className="text-xs font-semibold text-muted-foreground">
            Tidak ada riwayat kelainan yang cocok dengan kriteria filter.
          </p>
          {isFiltered && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleResetFilters}
              className="text-[10px] h-7"
            >
              Reset Filter
            </Button>
          )}
        </div>
      ) : (
        /* Slide Navigation Bar Eksklusif di Bawah Tabel */
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 px-5 py-3 border-t border-border/60 bg-secondary/30 text-xs">
          {/* Sisi Kiri: Rentang Data & Indikator Slide */}
          <div className="flex items-center gap-2.5 flex-wrap">
            <span className="text-[11px] text-muted-foreground font-mono">
              Menampilkan <b className="text-foreground">{startIndex + 1}–{endIndex}</b> dari <b className="text-foreground">{filtered.length}</b> riwayat
            </span>
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold bg-primary/10 text-primary border border-primary/20">
              Slide {currentSlide} dari {totalSlides}
            </span>
          </div>

          {/* Sisi Kanan: Pilihan Baris & Tombol Navigasi Slide */}
          <div className="flex items-center gap-3 flex-wrap">
            {/* Opsi Ukuran Baris per Slide */}
            <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
              <span>Per slide:</span>
              <select
                value={pageSize}
                onChange={(e) => setPageSize(Number(e.target.value))}
                className="h-7 rounded-md border bg-background px-2 text-[10px] font-semibold outline-none focus:ring-1 focus:ring-primary cursor-pointer text-foreground"
                title="Jumlah riwayat per slide (10 direkomendasikan agar pas di layar tanpa scroll)"
              >
                <option value={10}>10 (Pas Layar)</option>
                <option value={15}>15</option>
                <option value={25}>25</option>
                <option value={50}>50</option>
              </select>
            </div>

            {/* Tombol Paging Slide */}
            <div className="flex items-center gap-1">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentSlide(1)}
                disabled={currentSlide === 1}
                className="h-7 w-7 p-0 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                title="Ke Slide Pertama (Data Terbaru)"
              >
                <ChevronsLeft className="size-3.5" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentSlide((prev) => Math.max(1, prev - 1))}
                disabled={currentSlide === 1}
                className="h-7 px-2 text-[10px] gap-1 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                title="Slide Sebelumnya"
              >
                <ChevronLeft className="size-3" />
                <span className="hidden sm:inline">Prev</span>
              </Button>

              {/* Angka Slide Interaktif */}
              <div className="flex items-center gap-1">
                {getSlidePageNumbers(currentSlide, totalSlides).map((pageNum, idx) => {
                  if (pageNum === "...") {
                    return (
                      <span key={`ellipsis-${idx}`} className="px-1 text-[10px] text-muted-foreground">
                        …
                      </span>
                    );
                  }
                  const num = Number(pageNum);
                  const isActive = num === currentSlide;
                  return (
                    <button
                      key={`page-${num}`}
                      type="button"
                      onClick={() => setCurrentSlide(num)}
                      className={`h-7 min-w-[28px] px-1.5 rounded-md text-[10px] font-bold transition-all cursor-pointer ${
                        isActive
                          ? "bg-primary text-primary-foreground shadow-2xs"
                          : "hover:bg-secondary text-foreground"
                      }`}
                    >
                      {num}
                    </button>
                  );
                })}
              </div>

              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentSlide((prev) => Math.min(totalSlides, prev + 1))}
                disabled={currentSlide === totalSlides}
                className="h-7 px-2 text-[10px] gap-1 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                title="Slide Berikutnya (Data Lebih Lampau)"
              >
                <span className="hidden sm:inline">Next</span>
                <ChevronRight className="size-3" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentSlide(totalSlides)}
                disabled={currentSlide === totalSlides}
                className="h-7 w-7 p-0 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                title="Ke Slide Terakhir (Data Terlampau)"
              >
                <ChevronsRight className="size-3.5" />
              </Button>
            </div>
          </div>
        </div>
      )}
    </Panel>
  );
}

function Filter({
  value,
  setValue,
  label,
  options,
}: {
  value: string;
  setValue: (v: string) => void;
  label: string;
  options: (string | { value: string; label: string })[] | readonly string[];
}) {
  return (
    <label className="relative inline-block shrink-0">
      <span className="sr-only">{label}</span>
      <select
        value={value}
        onChange={(e) => setValue(e.target.value)}
        className="h-8 appearance-none rounded-lg border bg-secondary/70 py-1 pl-2.5 pr-6 text-[9.5px] font-semibold outline-none focus:ring-1 focus:ring-primary text-foreground transition-all hover:bg-secondary cursor-pointer"
      >
        <option value="ALL">{label}</option>
        {options.map((opt) => {
          const val = typeof opt === "string" ? opt : opt.value;
          const lbl = typeof opt === "string" ? opt : opt.label;
          return (
            <option key={val} value={val}>
              {lbl}
            </option>
          );
        })}
      </select>
      <ChevronDown className="pointer-events-none absolute top-2.5 right-1.5 size-3 text-muted-foreground" />
    </label>
  );
}

function RecordTable({
  rows,
  compact = false,
  onSelectRecord,
}: {
  rows: RecordRow[];
  compact?: boolean;
  onSelectRecord?: (record: RecordRow) => void;
}) {
  const cellPy = compact ? "py-2.5" : "py-3.5";
  const cellPx = compact ? "px-4" : "px-5";
  const headPy = compact ? "py-2.5" : "py-3";

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[950px] text-left">
        <thead className="bg-secondary/70 text-[9px] uppercase text-muted-foreground">
          <tr>
            <th className={`${cellPx} ${headPy} font-bold`}>Tanggal</th>
            <th className={`${cellPx} ${headPy} font-bold`}>Waktu</th>
            <th className={`${cellPx} ${headPy} font-bold`}>Abnormality ID</th>
            <th className={`${cellPx} ${headPy} font-bold`}>Kamera</th>
            <th className={`${cellPx} ${headPy} font-bold`}>Objek</th>
            <th className={`${cellPx} ${headPy} font-bold`}>Abnormality Type</th>
            <th className={`${cellPx} ${headPy} font-bold`}>AI Conf.</th>
            <th className={`${cellPx} ${headPy} font-bold`}>Severity</th>
            <th className={`${cellPx} ${headPy} font-bold`}>Status Lifecycle</th>
            <th className={`${cellPx} ${headPy} font-bold text-right`}>Aksi</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {rows.map((r) => {
            const dateInfo = extractRecordDate(r);
            return (
              <tr
                key={r.id + (r.detectionId ? `-${r.detectionId}` : "")}
                onClick={() => onSelectRecord?.(r)}
                className="group text-[10px] hover:bg-primary/5 transition-colors cursor-pointer"
                title="Klik untuk membuka bukti visual di Evidence Viewer"
              >
                {/* Kolom Tanggal (Date) */}
                <td className={`${cellPx} ${cellPy} whitespace-nowrap`}>
                  <div className="flex items-center gap-1.5 font-mono text-[10px] font-semibold text-foreground">
                    <Calendar className="size-3 text-primary/70 shrink-0" />
                    <span>{dateInfo.displayDate}</span>
                  </div>
                </td>

                {/* Kolom Waktu (Time) */}
                <td className={`${cellPx} ${cellPy} font-mono text-muted-foreground whitespace-nowrap`}>
                  {r.time}
                  {compact && " WIB"}
                </td>

                {/* Kolom Abnormality ID dengan Deep Link */}
                <td className={`${cellPx} ${cellPy} font-mono font-bold text-primary whitespace-nowrap`}>
                  <span className="inline-flex items-center gap-1 group-hover:underline">
                    {r.id}
                    <ExternalLink className="size-2.5 opacity-0 group-hover:opacity-100 transition-opacity text-primary" />
                  </span>
                </td>

                {/* Kamera */}
                <td className={`${cellPx} ${cellPy} font-bold whitespace-nowrap`}>{r.camera}</td>

                {/* Objek */}
                <td className={`${cellPx} ${cellPy} whitespace-nowrap`}>{r.object}</td>

                {/* Tipe Cacat */}
                <td
                  className={`${cellPx} ${cellPy} font-semibold whitespace-nowrap ${
                    r.severity === "CRITICAL"
                      ? "text-danger"
                      : r.severity === "HIGH"
                      ? "text-primary"
                      : r.severity === "MEDIUM"
                      ? "text-warning"
                      : ""
                  }`}
                >
                  <div>{r.type}</div>
                  {(r.pvFlatness != null || r.maxDeviation != null) && (
                    <div className="text-[8.5px] font-mono text-muted-foreground font-normal">
                      {r.pvFlatness != null && `PV: ${formatPxAndMm(r.pvFlatness)}`}
                      {r.pvFlatness != null && r.maxDeviation != null && " • "}
                      {r.maxDeviation != null && `Dev: ${formatPxAndMm(r.maxDeviation)}`}
                    </div>
                  )}
                </td>

                {/* AI Confidence */}
                <td className={`${cellPx} ${cellPy} font-mono font-bold whitespace-nowrap`}>{r.confidence}</td>

                {/* Severity */}
                <td className={`${cellPx} ${cellPy} whitespace-nowrap`}>
                  <Badge text={r.severity} />
                </td>

                {/* Status Lifecycle */}
                <td className={`${cellPx} ${cellPy} whitespace-nowrap`}>
                  <StatusBadge text={r.status} />
                </td>

                {/* Tombol Aksi: Lihat Bukti */}
                <td className={`${cellPx} ${cellPy} whitespace-nowrap text-right`}>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectRecord?.(r);
                    }}
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-primary/10 text-primary hover:bg-primary hover:text-white transition-all text-[9px] font-bold cursor-pointer shadow-2xs group-hover:scale-105"
                  >
                    <Eye className="size-3" />
                    <span>Lihat Bukti</span>
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Badge({ text }: { text: Severity }) {
  const cls: Record<Severity, string> = {
    CRITICAL: "border-danger/30 bg-danger-soft text-danger",
    HIGH: "border-primary/30 bg-info-soft text-primary",
    MEDIUM: "border-warning/40 bg-warning-soft text-warning",
    LOW: "border-border bg-secondary text-muted-foreground",
    NONE: "border-border bg-secondary text-muted-foreground",
  };
  return (
    <span
      className={`rounded border px-2 py-0.5 text-[8px] font-bold ${
        cls[text] || cls.LOW
      }`}
    >
      {text}
    </span>
  );
}

function StatusBadge({ text }: { text: string }) {
  const cls =
    text === "FALSE ALARM" || text === "INVALID"
      ? "border-muted-foreground/30 bg-secondary text-muted-foreground"
      : text === "RESOLVED"
      ? "border-success/30 bg-success-soft text-success"
      : text.startsWith("ACK")
      ? "border-info/30 bg-info-soft text-info"
      : "border-warning/40 bg-warning-soft text-warning";
  return (
    <span
      className={`whitespace-nowrap rounded border px-2 py-0.5 text-[8px] font-semibold ${cls}`}
    >
      {text}
    </span>
  );
}

const evidence: EvidenceItem[] = [
  {
    id: "ABN-20261020-00002",
    inspectionId: 218,
    camera: "CAM-01",
    title: "Plat Bending #00002",
    severity: "MEDIUM",
    object: "Plat (Slat #4)",
    date: "2026-10-20",
    displayDate: "20 Okt 2026",
    time: "10:10:44 WIB",
    area: "Conveyor Plate & Rib Overhead",
    confidence: "92.00%",
    statusLifecycle: "UNACKNOWLEDGED",
  },
  {
    id: "ABN-20261020-00001",
    inspectionId: 217,
    camera: "CAM-01",
    title: "Rib Bending #00001",
    severity: "MEDIUM",
    object: "Rib (Slat #4)",
    date: "2026-10-20",
    displayDate: "20 Okt 2026",
    time: "09:45:10 WIB",
    area: "Conveyor Plate & Rib Overhead",
    confidence: "95.00%",
    statusLifecycle: "UNACKNOWLEDGED",
  },
  {
    id: "ABN-20261019-00015",
    inspectionId: 195,
    camera: "CAM-01",
    title: "Rib Bending #00015",
    severity: "MEDIUM",
    object: "Rib (Slat #3)",
    date: "2026-10-19",
    displayDate: "19 Okt 2026",
    time: "15:10:43 WIB",
    area: "Conveyor Plate & Rib Overhead",
    confidence: "95.00%",
    statusLifecycle: "UNACKNOWLEDGED",
  },
  {
    id: "ABN-20261017-00008",
    inspectionId: 142,
    camera: "CAM-01",
    title: "Rib Bending #00008",
    severity: "MEDIUM",
    object: "Rib (Slat #53)",
    date: "2026-10-17",
    displayDate: "17 Okt 2026",
    time: "14:15:52 WIB",
    area: "Conveyor Plate & Rib Overhead",
    confidence: "95.00%",
    statusLifecycle: "UNACKNOWLEDGED",
  },
  {
    id: "ABN-20261016-00005",
    inspectionId: 110,
    camera: "CAM-01",
    title: "Rib Bending #00005",
    severity: "MEDIUM",
    object: "Rib (Slat #52)",
    date: "2026-10-16",
    displayDate: "16 Okt 2026",
    time: "11:15:50 WIB",
    area: "Conveyor Plate & Rib Overhead",
    confidence: "95.00%",
    statusLifecycle: "UNACKNOWLEDGED",
  },
  {
    id: "ABN-20261015-00003",
    inspectionId: 65,
    camera: "CAM-02",
    title: "Bolt Loose #00003",
    severity: "HIGH",
    object: "Fastener M16",
    date: "2026-10-15",
    displayDate: "15 Okt 2026",
    time: "10:28:44 WIB",
    area: "Bolt & Pin Right Side",
    confidence: "97.20%",
    statusLifecycle: "ACKNOWLEDGED",
  },
  {
    id: "ABN-20261014-00001",
    inspectionId: 24,
    camera: "CAM-03",
    title: "Pin Missing #00001",
    severity: "MEDIUM",
    object: "Connecting Pin #11",
    date: "2026-10-14",
    displayDate: "14 Okt 2026",
    time: "08:21:05 WIB",
    area: "Bolt & Pin Left Side",
    confidence: "91.40%",
    statusLifecycle: "RESOLVED",
  },
];

function extractPicRemarkOnly(picName: string | undefined, userName: string): string {
  if (!picName) return "";
  const prefix = `${userName} - `;
  if (picName.startsWith(prefix)) return picName.slice(prefix.length).trim();
  if (picName.trim() === userName.trim()) return "";
  const dashIdx = picName.indexOf(" - ");
  return dashIdx !== -1 ? picName.slice(dashIdx + 3).trim() : "";
}

function getEvidenceStatusLabel(item: EvidenceItem): string {
  if (item.overallResult === "INVALID" || item.detectionResult === "INVALID") {
    return "FALSE ALARM";
  }
  if (item.statusLifecycle === "RESOLVED") return "RESOLVED";
  if (item.statusLifecycle === "ACKNOWLEDGED") return "ACK";
  return "OPEN";
}

function Evidence({
  evidenceList,
  onRefresh,
  selectedTraceId,
  selectedCamera,
  onSelectTraceId,
  onSelectCamera,
  onBackToHistory,
}: {
  evidenceList: EvidenceItem[];
  onRefresh?: () => void;
  selectedTraceId?: string | null;
  selectedCamera?: string | null;
  onSelectTraceId?: (id: string | null) => void;
  onSelectCamera?: (cam: string | null) => void;
  onBackToHistory?: () => void;
}) {
  const currentItems = evidenceList.length > 0 ? evidenceList : evidence;

  const [evidenceTab, setEvidenceTab] = useState<"OPEN" | "RESOLVED" | "INVALID" | "ALL">("OPEN");
  const [actionFeedback, setActionFeedback] = useState<{
    type: "success" | "info" | "error";
    message: string;
    jumpTab?: "RESOLVED" | "INVALID";
  } | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [logs, setLogs] = useState<MaintenanceLogItem[]>([]);
  const lastMovedIdRef = React.useRef<string | null>(null);
  const lastAutoSyncedTraceIdRef = React.useRef<string | null>(null);

  const tabCounts = useMemo(() => {
    const counts = { OPEN: 0, RESOLVED: 0, INVALID: 0, ALL: currentItems.length };
    currentItems.forEach((item) => {
      counts[getRecordBucket(item)] += 1;
    });
    return counts;
  }, [currentItems]);

  const visibleItems = useMemo(() => {
    if (evidenceTab === "ALL") return currentItems;
    return currentItems.filter((item) => getRecordBucket(item) === evidenceTab);
  }, [currentItems, evidenceTab]);

  // Saat deep-link membawa selectedTraceId baru yang berbeda tab, arahkan tabnya SEKALI saja (tidak saat polling 4 detik)
  useEffect(() => {
    if (!selectedTraceId) {
      lastAutoSyncedTraceIdRef.current = null;
      return;
    }
    if (currentItems.length === 0) return;
    if (lastMovedIdRef.current === selectedTraceId) return;
    if (lastAutoSyncedTraceIdRef.current === selectedTraceId) return;

    const target = currentItems.find(
      (e) => e.id === selectedTraceId || e.inspectionId.toString() === selectedTraceId
    );
    if (target) {
      lastAutoSyncedTraceIdRef.current = selectedTraceId;
      const bucket = getRecordBucket(target);
      if (evidenceTab !== "ALL" && bucket !== evidenceTab) {
        setEvidenceTab(bucket);
      }
    }
  }, [selectedTraceId, currentItems, evidenceTab]);

  const handleSwitchEvidenceTab = (tabId: "OPEN" | "RESOLVED" | "INVALID" | "ALL") => {
    lastMovedIdRef.current = null;
    setEvidenceTab(tabId);

    const nextItems =
      tabId === "ALL"
        ? currentItems
        : currentItems.filter((item) => getRecordBucket(item) === tabId);

    const stillVisible = selectedTraceId
      ? nextItems.some((e) => e.id === selectedTraceId || e.inspectionId.toString() === selectedTraceId)
      : false;

    if (!stillVisible) {
      const nextItem = nextItems[0] || null;
      if (nextItem) {
        lastAutoSyncedTraceIdRef.current = nextItem.id;
        if (typeof window !== "undefined") {
          sessionStorage.setItem("last_valid_trace_id", nextItem.id);
        }
        onSelectTraceId?.(nextItem.id);
        if (nextItem.camera) onSelectCamera?.(nextItem.camera);
      } else {
        lastAutoSyncedTraceIdRef.current = selectedTraceId ?? null;
      }
    }
  };

  // Single Source of Truth: Item aktif langsung di-derive dari selectedTraceId, selectedCamera, dan visibleItems
  const selected = useMemo(() => {
    if (visibleItems.length === 0) return null;

    if (selectedTraceId) {
      let match = null;
      if (selectedCamera) {
        match = visibleItems.find(
          (e) =>
            (e.id === selectedTraceId || e.inspectionId.toString() === selectedTraceId) &&
            e.camera.toUpperCase() === selectedCamera.toUpperCase()
        );
      }
      if (!match) {
        match = visibleItems.find(
          (e) => e.id === selectedTraceId || e.inspectionId.toString() === selectedTraceId
        );
      }
      if (match) {
        if (typeof window !== "undefined" && evidenceList.length > 0) {
          sessionStorage.setItem("last_valid_trace_id", match.id);
        }
        return match;
      }
    }

    if (selectedCamera) {
      const camMatch = visibleItems.find(
        (e) => e.camera.toUpperCase() === selectedCamera.toUpperCase()
      );
      if (camMatch) return camMatch;
    }

    if (typeof window !== "undefined") {
      const lastValid = sessionStorage.getItem("last_valid_trace_id");
      if (lastValid) {
        const prevMatch = visibleItems.find(
          (e) => e.id === lastValid || e.inspectionId.toString() === lastValid
        );
        if (prevMatch) return prevMatch;
      }
    }

    return visibleItems[0] || null;
  }, [selectedTraceId, selectedCamera, visibleItems, evidenceList.length]);

  const [open, setOpen] = useState<Record<string, boolean>>({
    "CAM-01": selectedCamera ? selectedCamera.toUpperCase() === "CAM-01" : true,
    "CAM-02": selectedCamera ? selectedCamera.toUpperCase() === "CAM-02" : false,
    "CAM-03": selectedCamera ? selectedCamera.toUpperCase() === "CAM-03" : false,
  });
  const [openDates, setOpenDates] = useState<Record<string, boolean>>({});
  const [action, setAction] = useState(selected?.statusLifecycle || "UNACKNOWLEDGED");
  const [maintenanceAction, setMaintenanceAction] = useState(selected?.maintenanceNote || "");
  const [picRemark, setPicRemark] = useState("");
  const userName = (typeof window !== "undefined" && localStorage.getItem("userName")) || "Sarah Jenkins";

  // Sinkronkan form teknisi, folder explorer, & audit logs saat item terpilih berubah
  useEffect(() => {
    if (selected) {
      setAction(selected.statusLifecycle || "UNACKNOWLEDGED");
      setMaintenanceAction(selected.maintenanceNote || "");
      setPicRemark(extractPicRemarkOnly(selected.picName, userName));

      if (selected.camera) {
        setOpen((prev) => ({ ...prev, [selected.camera]: true }));
        const matchDate = extractEvidenceDate(selected).date;
        setOpenDates((prev) => ({
          ...prev,
          [`${selected.camera}_${matchDate}`]: true,
        }));
      }

      if (selected.inspectionId > 0) {
        fetchInspectionLogs(selected.inspectionId)
          .then(setLogs)
          .catch(() => setLogs([]));
      } else {
        setLogs([]);
      }
    } else {
      setLogs([]);
    }
  }, [selected?.id, selected?.inspectionId, selected?.statusLifecycle, selected?.overallResult, userName]);

  // Auto-Redirect URL: Jika parameter trace di URL tidak ditemukan di database sama sekali
  useEffect(() => {
    if (selectedTraceId && evidenceList.length > 0) {
      const match = evidenceList.find(
        (e) => e.id === selectedTraceId || e.inspectionId.toString() === selectedTraceId
      );
      if (!match && selected?.id && onSelectTraceId) {
        onSelectTraceId(selected.id);
      }
    }
  }, [selectedTraceId, evidenceList, selected?.id, onSelectTraceId]);

  const toggleDateFolder = (key: string, currentOpenState: boolean) => {
    setOpenDates((prev) => ({
      ...prev,
      [key]: !currentOpenState,
    }));
  };

  // Sinkronkan folder saat parameter kamera berubah di URL
  useEffect(() => {
    if (selectedCamera) {
      setOpen((prev) => ({ ...prev, [selectedCamera.toUpperCase()]: true }));
    }
  }, [selectedCamera]);

  const handleSelectItem = (item: EvidenceItem) => {
    lastMovedIdRef.current = null;
    setActionFeedback(null);
    if (typeof window !== "undefined") {
      sessionStorage.setItem("last_valid_trace_id", item.id);
    }
    if (onSelectTraceId) {
      onSelectTraceId(item.id);
    }
    if (onSelectCamera && item.camera) {
      onSelectCamera(item.camera);
    }
    setAction(item.statusLifecycle || "UNACKNOWLEDGED");
    setMaintenanceAction(item.maintenanceNote || "");
    setPicRemark(extractPicRemarkOnly(item.picName, userName));
  };

  const handleLifecycleAction = async (
    targetAction: "ACKNOWLEDGED" | "RESOLVED" | "FALSE_ALARM" | "UNACKNOWLEDGED"
  ) => {
    if (!selected || isSubmitting) return;
    setIsSubmitting(true);
    setActionFeedback(null);

    const defaultNotes: Record<typeof targetAction, string> = {
      ACKNOWLEDGED: "Dikonfirmasi oleh QA PIC",
      RESOLVED: "Telah diperbaiki & dikalibrasi ulang",
      FALSE_ALARM: "Ditandai sebagai False Alarm / Invalid oleh PIC",
      UNACKNOWLEDGED: "Dibuka kembali untuk pengecekan ulang",
    };

    const picLabel = picRemark.trim() ? `${userName} - ${picRemark.trim()}` : userName;
    const noteLabel = maintenanceAction.trim() || defaultNotes[targetAction];

    try {
      const movedTraceId = selected.id;
      const movedInspectionId = selected.inspectionId;
      await submitInspectionAction(movedInspectionId, targetAction, picLabel, noteLabel);

      if (targetAction === "RESOLVED" || targetAction === "FALSE_ALARM") {
        lastMovedIdRef.current = movedTraceId;
        if (evidenceTab === "OPEN") {
          const remainingOpen = visibleItems.filter((item) => item.id !== movedTraceId);
          const nextItem = remainingOpen[0] || null;
          if (typeof window !== "undefined") {
            if (nextItem) sessionStorage.setItem("last_valid_trace_id", nextItem.id);
            else sessionStorage.removeItem("last_valid_trace_id");
          }
          onSelectTraceId?.(nextItem ? nextItem.id : null);
          onSelectCamera?.(nextItem ? nextItem.camera : null);
        }
        const destTab = targetAction === "FALSE_ALARM" ? "INVALID" : "RESOLVED";
        setActionFeedback({
          type: "success",
          message:
            targetAction === "FALSE_ALARM"
              ? `${movedTraceId} ditandai sebagai False Alarm / Invalid dan dipindahkan dari antrian aktif.`
              : `${movedTraceId} ditandai RESOLVED (data tetap tersimpan) dan dipindahkan ke tab Resolved.`,
          jumpTab: destTab,
        });
      } else if (targetAction === "UNACKNOWLEDGED") {
        lastMovedIdRef.current = movedTraceId;
        setAction("UNACKNOWLEDGED");
        setEvidenceTab("OPEN");
        setActionFeedback({
          type: "info",
          message: `${movedTraceId} dibuka kembali ke antrian Active / Open.`,
        });
      } else {
        setAction("ACKNOWLEDGED");
        setActionFeedback({
          type: "info",
          message: `${movedTraceId} berhasil di-Acknowledge oleh ${userName}.`,
        });
      }

      if (movedInspectionId > 0) {
        fetchInspectionLogs(movedInspectionId).then(setLogs).catch(() => {});
      }
      if (onRefresh) onRefresh();
    } catch (e: any) {
      console.error(e);
      const detail = e?.message ? ` (${e.message})` : "";
      setActionFeedback({
        type: "error",
        message: `Gagal memperbarui status ke server backend${detail}. Pastikan server API Flask sudah di-restart.`,
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const cameraFolders = [
    {
      cam: "CAM-01",
      title: "Conveyor Plate & Rib",
      sub: "CAM-01 • Overhead Area",
      gradient: "bg-linear-to-r from-[#8b5cf6] to-[#6d28d9]",
      shadow: "shadow-md shadow-purple-500/20",
      subColor: "text-purple-100",
    },
    {
      cam: "CAM-02",
      title: "Bolt & Pin Right Side",
      sub: "CAM-02 • Right Oblique Side",
      gradient: "bg-linear-to-r from-[#3b82f6] to-[#1d4ed8]",
      shadow: "shadow-md shadow-blue-500/20",
      subColor: "text-blue-100",
    },
    {
      cam: "CAM-03",
      title: "Bolt & Pin Left Side",
      sub: "CAM-03 • Left Undercarriage Side",
      gradient: "bg-linear-to-r from-[#fb7185] to-[#e11d48]",
      shadow: "shadow-md shadow-rose-500/20",
      subColor: "text-rose-100",
    },
  ];

  const imgUrl = selected
    ? getImageUrl(selected.annotatedPath || selected.rawImagePath || selected.imageUrl)
    : null;
  const selectedBucket = selected ? getRecordBucket(selected) : "OPEN";
  const selectedStatusLabel = selected ? getEvidenceStatusLabel(selected) : "OPEN";

  const evidenceTabs: { id: "OPEN" | "RESOLVED" | "INVALID" | "ALL"; label: string; count: number }[] = [
    { id: "OPEN", label: "Active / Open", count: tabCounts.OPEN },
    { id: "RESOLVED", label: "Resolved", count: tabCounts.RESOLVED },
    { id: "INVALID", label: "False Alarm", count: tabCounts.INVALID },
    { id: "ALL", label: "Semua", count: tabCounts.ALL },
  ];

  return (
    <div className="space-y-4">
      {selectedTraceId && selected && (
        <div className="flex items-center justify-between rounded-xl border border-primary/30 bg-primary/10 px-4 py-2.5 text-xs text-primary shadow-xs">
          <div className="flex items-center gap-2">
            <Eye className="size-4 shrink-0" />
            <span>
              Menampilkan bukti rekaman visual untuk: <b className="font-mono">{selected.id}</b> ({selected.object} • {selected.camera})
            </span>
          </div>
          {onBackToHistory && (
            <button
              type="button"
              onClick={onBackToHistory}
              className="text-[10px] font-bold underline hover:text-foreground cursor-pointer"
            >
              Lihat di Daftar Histori &rarr;
            </button>
          )}
        </div>
      )}
      {actionFeedback && (
        <div
          className={`flex flex-wrap items-center justify-between gap-2 rounded-xl border px-4 py-2.5 text-xs shadow-xs ${
            actionFeedback.type === "success"
              ? "border-success/30 bg-success-soft text-success"
              : actionFeedback.type === "error"
              ? "border-danger/30 bg-danger-soft text-danger"
              : "border-info/30 bg-info-soft text-info"
          }`}
        >
          <span className="font-semibold">{actionFeedback.message}</span>
          <div className="flex items-center gap-2">
            {actionFeedback.jumpTab && (
              <button
                type="button"
                onClick={() => {
                  const targetTab = actionFeedback.jumpTab!;
                  const movedId = lastMovedIdRef.current;
                  lastAutoSyncedTraceIdRef.current = movedId;
                  setEvidenceTab(targetTab);
                  if (movedId) onSelectTraceId?.(movedId);
                  setActionFeedback(null);
                }}
                className="text-[10px] font-bold underline hover:opacity-80 cursor-pointer"
              >
                Lihat di Tab {actionFeedback.jumpTab === "INVALID" ? "False Alarm" : "Resolved"} &rarr;
              </button>
            )}
            <button
              type="button"
              onClick={() => setActionFeedback(null)}
              className="text-[10px] font-bold opacity-70 hover:opacity-100 cursor-pointer"
            >
              Tutup
            </button>
          </div>
        </div>
      )}
      <div className="grid gap-5 xl:grid-cols-[340px_1fr] items-start">
        <Panel className="overflow-hidden">
          <div className="border-b p-4 space-y-3">
            <div>
              <h3 className="text-xs font-extrabold">Camera Area Explorer</h3>
              <p className="text-[9px] text-muted-foreground">
                Klasifikasi Rekaman Anomali • {visibleItems.length} dari {currentItems.length} Files
              </p>
            </div>
            <div className="grid grid-cols-2 gap-1.5 rounded-xl bg-secondary/70 p-1 border border-border/60">
              {evidenceTabs.map((tab) => {
                const isActive = evidenceTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => handleSwitchEvidenceTab(tab.id)}
                    className={`flex items-center justify-between rounded-lg px-2 py-1.5 text-[9.5px] font-bold transition-all cursor-pointer ${
                      isActive
                        ? "bg-primary text-primary-foreground shadow-xs"
                        : "text-muted-foreground hover:bg-background/60 hover:text-foreground"
                    }`}
                  >
                    <span className="truncate">{tab.label}</span>
                    <span
                      className={`ml-1 rounded-full px-1.5 py-0.2 font-mono text-[8.5px] ${
                        isActive ? "bg-white/20 text-white" : "bg-background text-foreground"
                      }`}
                    >
                      {tab.count}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
          <div className="p-3 space-y-3">
            {cameraFolders.map((folder) => {
              const cam = folder.cam;
              const isOpen = open[cam];
              const items = visibleItems.filter((e) => e.camera === cam);
              return (
                <div key={cam} className="rounded-2xl border border-border/60 bg-secondary/30 p-1.5 transition-all">
                  <button
                    type="button"
                    onClick={() => setOpen({ ...open, [cam]: !isOpen })}
                    className={`w-full flex items-center justify-between p-3 rounded-xl ${folder.gradient} ${folder.shadow} text-white transition-all duration-200 hover:brightness-105 cursor-pointer`}
                  >
                    <div className="flex items-center gap-2.5 text-left">
                      <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-white/20 text-white backdrop-blur-xs shadow-xs">
                        {isOpen ? (
                          <FolderOpen className="size-4 text-white" />
                        ) : (
                          <Folder className="size-4 text-white" />
                        )}
                      </div>
                      <div>
                        <b className="block text-[11px] font-extrabold text-white leading-tight">
                          {folder.title}
                        </b>
                        <span className={`text-[8.5px] font-medium ${folder.subColor}`}>
                          {folder.sub}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="rounded-full bg-white/20 px-2 py-0.5 text-[8.5px] font-bold text-white">
                        {items.length} file
                      </span>
                      {isOpen ? (
                        <ChevronDown className="size-4 text-white" />
                      ) : (
                        <ChevronRight className="size-4 text-white" />
                      )}
                    </div>
                  </button>
                  {isOpen && (() => {
                    const dateGroupMap: Record<string, { date: string; displayDate: string; items: EvidenceItem[] }> = {};
                    items.forEach((item) => {
                      const d = extractEvidenceDate(item);
                      if (!dateGroupMap[d.date]) {
                        dateGroupMap[d.date] = {
                          date: d.date,
                          displayDate: d.displayDate,
                          items: [],
                        };
                      }
                      dateGroupMap[d.date].items.push(item);
                    });
                    const sortedDateGroups = Object.values(dateGroupMap).sort((a, b) => b.date.localeCompare(a.date));

                    return (
                      <div className="space-y-2 pt-2 px-1 pb-1 max-h-[420px] overflow-y-auto custom-scrollbar pr-1.5">
                        {items.length === 0 ? (
                          <div className="rounded-xl border border-dashed border-border/70 p-3 text-center text-[10px] text-muted-foreground">
                            Belum ada rekaman file pada filter ini
                          </div>
                        ) : (
                          sortedDateGroups.map((group, groupIdx) => {
                            const subfolderKey = `${cam}_${group.date}`;
                            const hasSelected = group.items.some((e) => e.id === selected?.id);
                            const isDateOpen = openDates[subfolderKey] !== undefined ? openDates[subfolderKey] : (groupIdx === 0 || hasSelected);

                            return (
                              <div
                                key={group.date}
                                className="rounded-xl border border-border/70 bg-card/60 p-1.5 transition-all shadow-2xs"
                              >
                                {/* Header Subfolder Tanggal */}
                                <button
                                  type="button"
                                  onClick={() => toggleDateFolder(subfolderKey, isDateOpen)}
                                  className="w-full flex items-center justify-between p-2 rounded-lg bg-secondary/50 hover:bg-secondary text-foreground transition-all cursor-pointer"
                                >
                                  <div className="flex items-center gap-2 text-left">
                                    <div className="flex size-6 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
                                      <Calendar className="size-3.5" />
                                    </div>
                                    <div>
                                      <div className="flex items-center gap-1.5">
                                        <b className="text-[10px] font-bold text-foreground">
                                          {group.displayDate}
                                        </b>
                                        {groupIdx === 0 && (
                                          <span className="rounded bg-primary/15 px-1.5 py-0.2 text-[7.5px] font-extrabold text-primary">
                                            Terbaru
                                          </span>
                                        )}
                                      </div>
                                      <span className="text-[8px] text-muted-foreground font-mono">
                                        {group.date}
                                      </span>
                                    </div>
                                  </div>
                                  <div className="flex items-center gap-1.5">
                                    <span className="rounded-full bg-secondary px-1.5 py-0.5 text-[8px] font-semibold text-muted-foreground border">
                                      {group.items.length} file
                                    </span>
                                    {isDateOpen ? (
                                      <ChevronDown className="size-3.5 text-muted-foreground" />
                                    ) : (
                                      <ChevronRight className="size-3.5 text-muted-foreground" />
                                    )}
                                  </div>
                                </button>

                                {/* Daftar File di dalam Subfolder Tanggal */}
                                {isDateOpen && (
                                  <div className="mt-1.5 space-y-1.5 pl-1 pr-0.5">
                                    {group.items.map((e) => {
                                      // Nomor urut harian selaras dengan Abnormality ID (misal #00002)
                                      const seq = e.id.split("-").pop() || `${e.inspectionId}`;
                                      const normalizedTitle = e.title.includes("#")
                                        ? e.title.replace(/#\d+/, `#${seq}`)
                                        : `${e.title} #${seq}`;
                                      const isSelected = !!selected && selected.id === e.id && (selected.object === e.object || selected.inspectionId === e.inspectionId);
                                      const itemStatusLabel = getEvidenceStatusLabel(e);

                                      return (
                                        <button
                                          key={e.id + (e.title || "")}
                                          type="button"
                                          onClick={() => handleSelectItem(e)}
                                          className={`w-full rounded-lg border p-2 text-left transition-all cursor-pointer ${
                                            isSelected
                                              ? "border-primary bg-primary/10 shadow-xs ring-1 ring-primary/40 font-semibold"
                                              : "border-border/60 bg-card hover:bg-secondary/70"
                                          }`}
                                        >
                                          <div className="flex justify-between items-center gap-1.5">
                                            <span
                                              className={`text-[9.5px] truncate ${
                                                isSelected ? "text-primary font-extrabold" : "font-bold text-foreground"
                                              }`}
                                            >
                                              {normalizedTitle}
                                            </span>
                                            <Badge text={e.severity} />
                                          </div>
                                          <div className="mt-1 flex justify-between items-center gap-1 text-[8px] text-muted-foreground">
                                            <span className="truncate max-w-[115px]">{e.object}</span>
                                            <div className="flex items-center gap-1 shrink-0">
                                              <StatusBadge text={itemStatusLabel} />
                                              <span className="font-mono">{e.time.replace(" WIB", "")}</span>
                                            </div>
                                          </div>
                                        </button>
                                      );
                                    })}
                                  </div>
                                )}
                              </div>
                            );
                          })
                        )}
                      </div>
                    );
                  })()}
                </div>
              );
            })}
          </div>
        </Panel>
        {!selected ? (
          <Panel className="p-10 text-center space-y-3">
            <CheckCircle2 className="mx-auto size-10 text-success" />
            <h3 className="text-sm font-extrabold">
              {evidenceTab === "OPEN"
                ? "Semua Anomali Aktif Telah Ditangani"
                : "Belum Ada Data pada Kategori Ini"}
            </h3>
            <p className="text-xs text-muted-foreground max-w-md mx-auto">
              {evidenceTab === "OPEN"
                ? "Tidak ada temuan NG berstatus Open / Acknowledged saat ini. Anda dapat meninjau arsip perbaikan atau false alarm di tab samping."
                : "Belum ada rekaman inspeksi pada tab filter yang sedang dipilih."}
            </p>
            <div className="flex flex-wrap justify-center gap-2 pt-2">
              {tabCounts.RESOLVED > 0 && evidenceTab !== "RESOLVED" && (
                <Button size="sm" variant="outline" onClick={() => handleSwitchEvidenceTab("RESOLVED")}>
                  Lihat Arsip Resolved ({tabCounts.RESOLVED})
                </Button>
              )}
              {tabCounts.INVALID > 0 && evidenceTab !== "INVALID" && (
                <Button size="sm" variant="outline" onClick={() => handleSwitchEvidenceTab("INVALID")}>
                  Lihat False Alarm ({tabCounts.INVALID})
                </Button>
              )}
              {evidenceTab !== "ALL" && tabCounts.ALL > 0 && (
                <Button size="sm" variant="ghost" onClick={() => handleSwitchEvidenceTab("ALL")}>
                  Tampilkan Semua ({tabCounts.ALL})
                </Button>
              )}
            </div>
          </Panel>
        ) : (
        <div className="space-y-5">
          <Panel className="overflow-hidden">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b p-4">
              <div>
                <h3 className="text-xs font-extrabold flex items-center gap-2">
                  <span>RAW Evidence Optical Capture • {selected.camera}</span>
                  <span className="rounded bg-primary/10 px-2 py-0.5 text-[9px] font-bold text-primary font-mono">
                    {(() => {
                      const seq = selected.id.split("-").pop() || `${selected.inspectionId}`;
                      return selected.title.includes("#")
                        ? selected.title.replace(/#\d+/, `#${seq}`)
                        : `${selected.title} #${seq}`;
                    })()}
                  </span>
                </h3>
                <p className="mt-1 font-mono text-[9px] text-muted-foreground">
                  {selected.id} • {selected.captureTime || selected.time}
                </p>
              </div>
              <Badge text={selected.severity} />
            </div>
            <div className="camera-grid relative flex w-full aspect-video items-center justify-center overflow-hidden bg-black/95">
              {imgUrl ? (
                <img
                  src={imgUrl}
                  alt={selected.title}
                  className="size-full object-contain"
                  onError={() => {
                    console.error("Gagal memuat gambar bukti:", imgUrl);
                  }}
                />
              ) : (
                <div className="h-2/5 w-1/3 border-2 border-danger flex items-center justify-center">
                  <span className="-mt-6 block w-fit bg-danger px-2 py-1 font-mono text-[9px] font-bold text-primary-foreground">
                    {(() => {
                      const seq = selected.id.split("-").pop() || `${selected.inspectionId}`;
                      const cleanT = selected.title.includes("#")
                        ? selected.title.replace(/#\d+/, `#${seq}`)
                        : `${selected.title} #${seq}`;
                      return cleanT.toUpperCase();
                    })()} • {selected.confidence} CONF
                  </span>
                </div>
              )}
              <span className="absolute bottom-4 left-4 rounded bg-black/80 px-2 py-1 font-mono text-[8px] text-white">
                {selected.annotatedPath ? "AI ANNOTATED OVERLAY" : "RAW OPTICAL EVIDENCE"}
              </span>
            </div>
          </Panel>
          <Panel className="p-5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <p className="text-[9px] font-bold uppercase text-muted-foreground">
                  Traceability ID
                </p>
                <h2 className="mt-1 font-mono text-base font-extrabold text-primary">
                  {selected.id}
                </h2>
              </div>
              <div className="flex items-center gap-2">
                {selected.picName && (
                  <span className="text-[9.5px] text-muted-foreground font-medium">
                    PIC: <b className="text-foreground">{selected.picName}</b>
                  </span>
                )}
                <StatusBadge text={selectedStatusLabel === "OPEN" ? action : selectedStatusLabel} />
              </div>
            </div>
            <dl className="mt-4 grid gap-4 border-y py-4 text-[10px] sm:grid-cols-2 lg:grid-cols-3">
              {[
                ["Target Camera", `${selected.camera} (${selected.area})`],
                ["Target Area", selected.area],
                ["Object", selected.object],
                ["Abnormality Type", selected.title.split(" #")[0] ?? ""],
                ["AI Confidence", selected.confidence],
                ["Capture Time", selected.captureTime || selected.time],
                ...(selected.resolvedAtDisplay
                  ? [[selectedBucket === "INVALID" ? "Waktu Ditandai Invalid" : "Waktu Resolved", selected.resolvedAtDisplay]]
                  : []),
                ...(selected.pvFlatness !== undefined && selected.pvFlatness !== null
                  ? [["Metrologi Flatness PV", formatPxAndMm(selected.pvFlatness)]]
                  : []),
                ...(selected.maxDeviation !== undefined && selected.maxDeviation !== null
                  ? [["Max Deviasi Defleksi", formatPxAndMm(selected.maxDeviation)]]
                  : []),
              ].map(([k, v]) => (
                <div key={k}>
                  <dt className="text-muted-foreground">{k}</dt>
                  <dd className="mt-1 font-bold">{v}</dd>
                </div>
              ))}
            </dl>
            <div className="mt-4 grid gap-4 sm:grid-cols-2">
              <label className="block text-[10px] font-bold">
                Tindakan Maintenance / Alasan Verifikasi
                <textarea
                  value={maintenanceAction}
                  onChange={(e) => setMaintenanceAction(e.target.value)}
                  className="mt-2 min-h-20 w-full resize-none rounded-lg border bg-secondary p-3 text-[10px] outline-none focus:ring-1 focus:ring-ring text-foreground"
                  placeholder="Deskripsi perbaikan atau alasan jika ini false alarm..."
                />
              </label>
              <label className="block text-[10px] font-bold">
                PIC Remark (Opsional)
                <textarea
                  value={picRemark}
                  onChange={(e) => setPicRemark(e.target.value)}
                  className="mt-2 min-h-20 w-full resize-none rounded-lg border bg-secondary p-3 text-[10px] outline-none focus:ring-1 focus:ring-ring text-foreground"
                  placeholder="Catatan tambahan PIC / shift teknisi..."
                />
              </label>
            </div>
            <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap gap-2">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => handleLifecycleAction("ACKNOWLEDGED")}
                  disabled={isSubmitting || selectedBucket !== "OPEN" || action === "ACKNOWLEDGED"}
                >
                  <Wrench className="size-4" />
                  Acknowledge
                </Button>
                <Button
                  size="sm"
                  className="bg-success text-white hover:bg-success/90"
                  onClick={() => handleLifecycleAction("RESOLVED")}
                  disabled={isSubmitting || selectedBucket === "RESOLVED"}
                >
                  <Check className="size-4" />
                  Mark Resolved
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="border-amber-500/40 text-amber-600 dark:text-amber-400 hover:bg-amber-500/10"
                  onClick={() => handleLifecycleAction("FALSE_ALARM")}
                  disabled={isSubmitting || selectedBucket === "INVALID"}
                  title="Tandai sebagai salah deteksi AI (False Positive) agar tidak masuk statistik NG"
                >
                  <Ban className="size-4" />
                  False Alarm / Invalid
                </Button>
              </div>
              {selectedBucket !== "OPEN" && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleLifecycleAction("UNACKNOWLEDGED")}
                  disabled={isSubmitting}
                  className="text-[10px] text-muted-foreground hover:text-foreground"
                  title="Kembalikan status item ini ke antrian Active / Open"
                >
                  <RotateCcw className="size-3.5" />
                  Reopen ke Active
                </Button>
              )}
            </div>
            {actionFeedback && (
              <div
                className={`mt-3 rounded-lg border px-3 py-2 text-[10px] font-semibold ${
                  actionFeedback.type === "success"
                    ? "border-success/30 bg-success-soft text-success"
                    : actionFeedback.type === "error"
                    ? "border-danger/30 bg-danger-soft text-danger"
                    : "border-info/30 bg-info-soft text-info"
                }`}
              >
                {actionFeedback.message}
              </div>
            )}

            {logs.length > 0 && (
              <div className="mt-5 border-t pt-4">
                <h4 className="flex items-center gap-1.5 text-[10px] font-extrabold uppercase tracking-wide text-muted-foreground">
                  <History className="size-3.5" />
                  Audit Trail Tindakan (Maintenance Logs)
                </h4>
                <div className="mt-2.5 space-y-2 max-h-40 overflow-y-auto custom-scrollbar pr-1">
                  {logs.map((log) => (
                    <div
                      key={log.logId}
                      className="flex flex-wrap items-start justify-between gap-2 rounded-lg border border-border/60 bg-secondary/40 px-3 py-2 text-[9.5px]"
                    >
                      <div className="space-y-0.5">
                        <div className="flex items-center gap-1.5">
                          <StatusBadge
                            text={log.isFalseAlarm ? "FALSE ALARM" : log.action}
                          />
                          <span className="font-bold text-foreground">{log.picName}</span>
                        </div>
                        {log.notes && (
                          <p className="text-muted-foreground">{log.notes}</p>
                        )}
                      </div>
                      <span className="font-mono text-[8.5px] text-muted-foreground shrink-0">
                        {log.loggedAtDisplay}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </Panel>
        </div>
        )}
      </div>
    </div>
  );
}

function Analytics({ data }: { data?: AnalyticsData | null }) {
  const defaultDays = [
    { d: "Sen", n: 4 },
    { d: "Sel", n: 7 },
    { d: "Rab", n: 3 },
    { d: "Kam", n: 9 },
    { d: "Jum", n: 12 },
    { d: "Sab", n: 5 },
    { d: "Min", n: 2 },
  ];

  const defaultPareto = [
    {
      label: "Rib Deformation (CAM-01)",
      n: 32,
      p: "42.6%",
      barCls: "bg-danger",
    },
    {
      label: "Bolt Missing / Loose (CAM-02)",
      n: 21,
      p: "28.0%",
      barCls: "bg-primary",
    },
    {
      label: "Pin Misalignment (CAM-03)",
      n: 14,
      p: "18.6%",
      barCls: "bg-warning",
    },
    {
      label: "Plate Scuff / Surface Wear",
      n: 8,
      p: "10.6%",
      barCls: "bg-info",
    },
  ];

  const days = data?.weekly && data.weekly.length > 0 ? data.weekly : defaultDays;
  const pareto = data?.pareto && data.pareto.length > 0 ? data.pareto : defaultPareto;
  const maxN = Math.max(...days.map((x) => x.n), 1);
  const resSummary = data?.resolutionSummary;

  return (
    <div className="space-y-5">
      {resSummary && (
        <div className="grid gap-3.5 sm:grid-cols-2 lg:grid-cols-4">
          <Panel className="p-4">
            <p className="text-[10px] font-bold text-muted-foreground">Active / Open NG</p>
            <p className="mt-1 text-2xl font-extrabold text-warning">{resSummary.openCount}</p>
            <p className="mt-1 text-[9px] text-muted-foreground">Menunggu tindakan maintenance</p>
          </Panel>
          <Panel className="p-4">
            <p className="text-[10px] font-bold text-muted-foreground">Resolved NG</p>
            <div className="mt-1 flex items-baseline gap-2">
              <p className="text-2xl font-extrabold text-success">{resSummary.resolvedCount}</p>
              <span className="text-[10px] font-mono font-bold text-success">
                ({resSummary.resolutionRatePct}%)
              </span>
            </div>
            <p className="mt-1 text-[9px] text-muted-foreground">Rasio penyelesaian kerusakan nyata</p>
          </Panel>
          <Panel className="p-4">
            <p className="text-[10px] font-bold text-muted-foreground">Rata-rata MTTR</p>
            <p className="mt-1 text-2xl font-extrabold text-primary">
              {resSummary.avgMttrMinutes !== null ? `${resSummary.avgMttrMinutes} mnt` : "-"}
            </p>
            <p className="mt-1 text-[9px] text-muted-foreground">Durasi deteksi hingga resolved</p>
          </Panel>
          <Panel className="p-4">
            <p className="text-[10px] font-bold text-muted-foreground">False Alarm / Invalid</p>
            <p className="mt-1 text-2xl font-extrabold text-foreground">{resSummary.invalidCount}</p>
            <p className="mt-1 text-[9px] text-muted-foreground">Dikecualikan dari trend & pareto</p>
          </Panel>
        </div>
      )}
      <div className="grid gap-5 xl:grid-cols-2">
      <Panel className="p-5">
        <SectionTitle
          title="Weekly Abnormality Trend"
          subtitle="7 Hari Terakhir"
        />
        <div className="mt-8 flex h-60 items-end justify-around gap-2 border-b border-l px-4">
          {days.map((item) => (
            <div
              key={item.d}
              className="flex h-full flex-1 flex-col items-center justify-end"
            >
              <span className="mb-2 text-xs font-bold">{item.n}</span>
              <div
                className="w-full max-w-10 rounded-t bg-primary/80 transition hover:bg-primary"
                style={{ height: `${(item.n / maxN) * 75}%` }}
              />
              <span className="my-3 text-[9px] text-muted-foreground">
                {item.d}
              </span>
            </div>
          ))}
        </div>
      </Panel>
      <Panel className="p-5">
        <SectionTitle
          title="Pareto Abnormality by Defect Type"
          subtitle={`Total Insiden Terdeteksi: ${data?.totalIncidents ?? 75}`}
        />
        <div className="mt-7 space-y-6">
          {pareto.map((item) => (
            <div key={item.label}>
              <div className="mb-2 flex justify-between text-[10px]">
                <b>{item.label}</b>
                <span className="font-mono font-bold">
                  {item.n} ({item.p})
                </span>
              </div>
              <div className="h-3 overflow-hidden rounded-full bg-secondary">
                <div
                  className={`h-full rounded-full ${item.barCls}`}
                  style={{ width: item.p }}
                />
              </div>
            </div>
          ))}
        </div>
      </Panel>
      </div>
    </div>
  );
}
