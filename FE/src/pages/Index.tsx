import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  AlertTriangle,
  BarChart3,
  Calendar,
  Camera,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Cpu,
  Download,
  ExternalLink,
  Eye,
  Folder,
  FolderOpen,
  Gauge,
  Layers,
  LayoutDashboard,
  LogOut,
  Moon,
  RotateCcw,
  Search,
  ShieldAlert,
  ShieldCheck,
  Sun,
  Wrench,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  fetchHealth,
  fetchStats,
  fetchInspections,
  fetchEvidence,
  submitInspectionAction,
  fetchAnalytics,
  getImageUrl,
  extractRecordDate,
  extractEvidenceDate,
  MONTH_NAMES_ID,
  MONTH_SHORT_ID,
  type RecordRow,
  type EvidenceItem,
  type AbnormalComponentSummary,
  type AnalyticsData,
  type Severity,
} from "@/lib/api";

type View = "dashboard" | "live" | "abnormality" | "evidence" | "analytics";

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
    date: "2026-09-13",
    displayDate: "13 Sep 2026",
    time: "10:31:22",
    id: "ABN-20260913-00021",
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
    date: "2026-09-13",
    displayDate: "13 Sep 2026",
    time: "10:28:44",
    id: "ABN-20260913-00020",
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
    date: "2026-09-13",
    displayDate: "13 Sep 2026",
    time: "10:21:05",
    id: "ABN-20260913-00019",
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
    date: "2026-09-13",
    displayDate: "13 Sep 2026",
    time: "10:14:18",
    id: "ABN-20260913-00018",
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
];

const titles: Record<View, [string, string]> = {
  dashboard: [
    "AI Vision Inspection Dashboard",
    "Real-time AI Vision Inspection & Traceability System",
  ],
  live: [
    "Live Camera Monitoring",
    "Synchronous optical stream from 3 target areas",
  ],
  abnormality: [
    "Abnormality History & Traceability Ledger",
    "Real-time AI Vision Inspection & Traceability System",
  ],
  evidence: [
    "Evidence Viewer",
    "Archived optical evidence and maintenance traceability",
  ],
  analytics: [
    "Analytics & Maintenance Insights",
    "Recurring abnormality trends and inspection performance",
  ],
};

export default function Index() {
  const [view, setView] = useState<View>("dashboard");
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | null>(null);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [apiStatus, setApiStatus] = useState<{ status: string; database: string }>({
    status: "checking",
    database: "checking",
  });
  const [statsData, setStatsData] = useState<AbnormalComponentSummary[]>(abnormalComponentCounts);
  const [recordsData, setRecordsData] = useState<RecordRow[]>(records);
  const [evidenceData, setEvidenceData] = useState<EvidenceItem[]>(evidence);
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null);

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
    setSelectedEvidenceId(record.id);
    setView("evidence");
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
        if (an) setAnalyticsData(an);
      }
    } catch (e) {
      console.error("Gagal sinkronisasi data:", e);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 4000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground lg:h-screen lg:overflow-hidden transition-colors duration-300">
      <Sidebar
        view={view}
        setView={setView}
        isCollapsed={isCollapsed}
        setIsCollapsed={setIsCollapsed}
        theme={theme}
        onToggleTheme={toggleTheme}
      />
      <div
        className={`transition-all duration-300 ease-in-out lg:flex lg:h-screen lg:flex-col ${
          isCollapsed ? "lg:ml-24" : "lg:ml-72"
        }`}
      >
        <Header
          view={view}
          apiStatus={apiStatus}
          theme={theme}
          onToggleTheme={toggleTheme}
        />
        <main className="flex-1 overflow-y-auto px-4 py-5 sm:px-6 lg:px-8 lg:py-7">
          {view === "dashboard" && (
            <Dashboard
              onViewAll={() => setView("abnormality")}
              onSelectRecord={handleSelectRecord}
              statsList={statsData}
              recentRecords={recordsData}
            />
          )}
          {view === "live" && <LiveCamera />}
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
              initialSelectedId={selectedEvidenceId}
              onBackToHistory={() => setView("abnormality")}
            />
          )}
          {view === "analytics" && <Analytics data={analyticsData} />}
        </main>
      </div>
    </div>
  );
}

function Sidebar({
  view,
  setView,
  isCollapsed,
  setIsCollapsed,
  theme,
  onToggleTheme,
}: {
  view: View;
  setView: (view: View) => void;
  isCollapsed: boolean;
  setIsCollapsed: React.Dispatch<React.SetStateAction<boolean>>;
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

  return (
    <aside
      className={`fixed inset-y-3 left-3 z-30 flex flex-col justify-between rounded-3xl transition-all duration-300 ease-in-out ${
        isDark
          ? "sidebar-gradient-emerald border border-emerald-400/40 text-white shadow-2xl shadow-emerald-950/60"
          : "bg-linear-to-b from-[#7645f7] via-[#6a37eb] to-[#5a24e2] border border-white/20 text-white shadow-2xl shadow-purple-900/30"
      } ${isCollapsed ? "w-20 py-5 px-2 items-center" : "w-64 py-5 px-4"}`}
    >
      <div>
        {isCollapsed ? (
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
          <div className="mb-6 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div
                className={`flex size-10 items-center justify-center rounded-2xl shadow-sm ${
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
              className={`flex size-8 shrink-0 items-center justify-center rounded-lg transition-colors cursor-pointer ${
                isDark
                  ? "bg-black/15 text-slate-950 hover:bg-black/25"
                  : "bg-white/10 text-white/80 hover:bg-white/20 hover:text-white"
              }`}
              title="Perkecil Side Bar"
            >
              <ChevronLeft className="size-4" />
            </button>
          </div>
        )}

        {/* Navigation Items */}
        <nav
          className={`mt-4 flex ${
            isCollapsed
              ? "flex-col items-center gap-2.5"
              : "flex-col space-y-1.5"
          }`}
          aria-label="Main navigation"
        >
          {nav.map((item, idx) => {
            const Icon = item.icon;
            const active = view === item.id;

            // Di tema gelap (gradient hijau):
            // - Bagian atas (idx < 2): berlatar neon hijau -> teks gelap
            // - Bagian bawah (idx >= 2): berlatar teal/charcoal -> teks terang / neon pill
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
              // Tema terang: ungu klasik seperti sebelumnya
              activeClass = isCollapsed
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
                  isCollapsed
                    ? `size-11 justify-center rounded-2xl ${
                        active ? activeClass : inactiveClass
                      }`
                    : `h-11 w-full gap-3 rounded-xl px-3.5 text-xs ${
                        active ? activeClass : inactiveClass
                      }`
                }`}
              >
                <Icon
                  className={`${isCollapsed ? "size-5" : "size-4.5"} shrink-0`}
                />
                {!isCollapsed && <span className="truncate">{item.label}</span>}
                {isCollapsed && (
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
      {isCollapsed ? (
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
  );
}

function Header({
  view,
  apiStatus,
  theme,
  onToggleTheme,
}: {
  view: View;
  apiStatus: { status: string; database: string };
  theme: "light" | "dark";
  onToggleTheme: () => void;
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
    <header className="flex items-center justify-between px-4 pt-5 pb-1 sm:px-6 lg:px-8 bg-transparent">
      <div>
        <h1 className="text-base font-extrabold sm:text-lg tracking-tight">
          {titles[view][0]}
        </h1>
        <p className="hidden text-[10px] text-muted-foreground sm:block">
          {titles[view][1]}
        </p>
      </div>
      <div className="flex items-center gap-2 sm:gap-3">
        <div className="hidden items-center gap-2 sm:flex">
          <StatusPill text={isOnline ? "API: Online" : "API: Offline"} online={isOnline} />
          <StatusPill text={isDb ? "SSMS: Connected" : "SSMS: Disconnected"} online={isDb} />
          <StatusPill text="3 Cams: Online" online={true} />
          <span className="flex items-center gap-1.5 rounded-lg border border-border/60 bg-card px-2.5 py-1.5 font-mono text-[9px] font-semibold text-muted-foreground shadow-xs">
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
}: {
  onViewAll: () => void;
  onSelectRecord?: (record: RecordRow) => void;
  statsList: AbnormalComponentSummary[];
  recentRecords: RecordRow[];
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
    return {
      label: s.label,
      value: s.count.toString(),
      camera: s.camera,
      note: meta.note,
      Icon: meta.Icon,
      tone: meta.tone,
      bgTone: meta.bgTone,
      dotColor: s.dotColor,
    };
  });

  return (
    <div className="space-y-6">
      <div className="grid gap-3.5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
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
                  <p className="text-2xl font-extrabold text-foreground">{stat.value}</p>
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
              1080p @ 30 FPS • Synchronous Ingest
            </span>
          }
        />
        <div className="mt-4 grid gap-4 xl:grid-cols-3">
          <CameraFeed
            camera="CAM-01"
            area="Plate / Rib"
            object="SLAT #4928 INTERSECT"
            objectType="RIB"
            conf="96.8%"
            fps="30"
          />
          <CameraFeed
            camera="CAM-02"
            area="Bolt / Pin RH Side"
            object="BOLT M16 SECURED"
            objectType="BOLT"
            conf="98.1%"
            fps="30"
          />
          <CameraFeed
            camera="CAM-03"
            area="Bolt / Pin LH Side"
            object="PIN DISLODGED / MISSING"
            objectType="PIN MISSING"
            conf="95.2%"
            fps="30"
            sev="HIGH"
            alert
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
          rows={recentRecords.slice(0, 5)}
          compact
          onSelectRecord={onSelectRecord}
        />
      </Panel>
    </div>
  );
}

function CameraFeed({
  camera,
  area,
  object,
  objectType,
  conf,
  fps = "30",
  sev,
  alert = false,
}: {
  camera: string;
  area: string;
  object: string;
  objectType: string;
  conf: string;
  fps?: string;
  sev?: string;
  alert?: boolean;
}) {
  return (
    <div
      className={`overflow-hidden rounded-lg border ${
        alert ? "border-danger/50" : "border-border"
      }`}
    >
      <div className="flex items-center justify-between bg-secondary px-3 py-2 text-[10px]">
        <span className="font-bold">
          {camera}{" "}
          <span className="font-medium text-muted-foreground">({area})</span>
        </span>
        <span className={`font-bold ${alert ? "text-danger" : "text-success"}`}>
          {alert ? "ALARM ACTIVE" : "ONLINE"}
        </span>
      </div>
      <div className="camera-grid relative flex aspect-video items-center justify-center overflow-hidden">
        <div
          className={`absolute inset-x-0 top-0 h-px ${
            alert ? "bg-danger" : "bg-success"
          } scan-line`}
        />
        <div
          className={`h-20 w-32 border-2 ${
            alert ? "border-danger" : "border-success"
          }`}
        >
          <span
            className={`-mt-5 block w-fit px-1 font-mono text-[8px] ${
              alert
                ? "bg-danger text-primary-foreground"
                : "bg-success text-primary-foreground"
            }`}
          >
            {object}
          </span>
        </div>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-1.5 bg-card px-3 py-2 font-mono text-[9px] text-muted-foreground border-t">
        <div className="flex items-center gap-1.5 sm:gap-2">
          <span>
            Object: <strong className="text-foreground">{objectType}</strong>
          </span>
          <span className="text-border">|</span>
          <span>
            Conf: <strong className="text-foreground">{conf}</strong>
          </span>
          <span className="text-border">|</span>
          <span className="flex items-center gap-1 font-bold text-foreground">
            <span className="size-1.5 rounded-full bg-success animate-pulse" />
            FPS: <strong className="text-foreground">{fps}</strong>
          </span>
        </div>
        {sev && (
          <span className="rounded border border-danger/30 bg-danger-soft px-1.5 py-0.5 text-[8px] font-bold text-danger">
            Sev: {sev}
          </span>
        )}
      </div>
    </div>
  );
}

function LiveCamera() {
  const cameras = [
    {
      cam: "CAM-01",
      status: "ONLINE",
      station: "Station: ST-01 / Overhead",
      title: "Plat / Rib Inspection Area",
      desc: "Mendeteksi deformasi profil rib conveyor.",
      obj: "Object: Slat Rib #4928",
      alert: false,
    },
    {
      cam: "CAM-02",
      status: "ONLINE",
      station: "Station: ST-01 / Right Oblique",
      title: "Bolt / Pin Right Side Area",
      desc: "Inspeksi baut penahan samping dan keberadaan washer flange.",
      obj: "Object: Fastener M16",
      alert: false,
    },
    {
      cam: "CAM-03",
      status: "ALARM",
      station: "Station: ST-01 / Left Undercarriage",
      title: "Bolt / Pin Left Side Area",
      desc: "Terdeteksi pin penghubung rantai atau pin hilang.",
      obj: "Pin Missing On Link #11",
      alert: true,
    },
  ];

  return (
    <div className="space-y-5">
      <SectionTitle
        title="Dedicated 3 Camera Industrial Feed"
        subtitle="Monitoring visual individual untuk tiap sudut kamera conveyor"
        aside={
          <div className="flex gap-2">
            <StatusPill text="Bounding Box: ON" />
            <StatusPill text="Shutter Sync: 100%" />
          </div>
        }
      />
      <div className="grid gap-5 xl:grid-cols-3">
        {cameras.map((c) => (
          <Panel key={c.cam} className="overflow-hidden">
            <div className="flex justify-between p-4 pb-3">
              <div>
                <p className="text-xs font-extrabold">
                  {c.cam}{" "}
                  <span className={c.alert ? "text-danger" : "text-success"}>
                    {c.status}
                  </span>
                </p>
                <p className="mt-1 text-[9px] text-muted-foreground">
                  {c.station}
                </p>
              </div>
              <Camera className="size-5 text-muted-foreground" />
            </div>
            <div className="camera-grid relative flex aspect-4/3 items-center justify-center overflow-hidden">
              <div
                className={`h-32 w-40 border-2 ${
                  c.alert ? "border-danger" : "border-success"
                }`}
              />
              <div
                className={`absolute bottom-3 left-3 rounded px-2 py-1 font-mono text-[9px] ${
                  c.alert ? "bg-danger" : "bg-success"
                } text-primary-foreground`}
              >
                {c.alert ? "CRITICAL DEFECT DETECTED" : "STREAM VIEW • OK"}
              </div>
            </div>
            <div className="p-4">
              <h3 className="text-sm font-bold">{c.title}</h3>
              <p className="mt-1 text-[10px] text-muted-foreground">{c.desc}</p>
              <div className="mt-4 flex justify-between border-t pt-3 text-[9px] font-mono">
                <span>
                  FPS: <b>30</b> • 1080p RAW
                </span>
                <b className={c.alert ? "text-danger" : "text-success"}>
                  {c.obj}
                </b>
              </div>
            </div>
          </Panel>
        ))}
      </div>
    </div>
  );
}

function Abnormality({
  allRecords,
  onSelectRecord,
}: {
  allRecords: RecordRow[];
  onSelectRecord?: (record: RecordRow) => void;
}) {
  const [camera, setCamera] = useState("ALL");
  const [severity, setSeverity] = useState("ALL");
  const [status, setStatus] = useState("ALL");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedYear, setSelectedYear] = useState("ALL");
  const [selectedMonth, setSelectedMonth] = useState("ALL");
  const [selectedDate, setSelectedDate] = useState("");

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
      if (camera !== "ALL" && r.camera !== camera) return false;
      if (severity !== "ALL" && r.severity !== severity) return false;
      if (
        status !== "ALL" &&
        !r.status.startsWith(status) &&
        r.statusLifecycle !== status
      )
        return false;

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
  }, [allRecords, camera, severity, status, selectedYear, selectedMonth, selectedDate, searchQuery]);

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
      "Tanggal,Waktu,Abnormality ID,Kamera,Objek,Tipe Kelainan,AI Confidence,Severity,Status,Timestamp Penuh\n";
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

  return (
    <Panel className="overflow-hidden">
      <div className="p-5 pb-4 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <SectionTitle
            title="Abnormality History & Traceability Ledger"
            subtitle="Pencatatan seluruh histori kelainan dengan kemampuan filter multi-parameter"
          />
          <div className="flex items-center gap-2 shrink-0">
            <span className="text-[10px] font-mono text-muted-foreground hidden lg:inline">
              Total: <b className="text-foreground">{filtered.length}</b> dari {allRecords.length}
            </span>
            <Button size="sm" onClick={exportCsv} className="h-8 text-xs gap-1.5 cursor-pointer">
              <Download className="size-3.5" />
              Export CSV
            </Button>
          </div>
        </div>

        {/* Filter Bar Multi-Parameter */}
        <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-border/50">
          {/* 1. Pencarian Kata Kunci */}
          <div className="relative inline-flex items-center min-w-[200px] flex-1 sm:flex-initial">
            <Search className="absolute left-2.5 pointer-events-none size-3 text-muted-foreground" />
            <input
              type="text"
              placeholder="Cari ID, objek, tipe..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="h-8 w-full rounded-lg border bg-secondary/70 py-1 pl-7.5 pr-3 text-[10px] placeholder:text-muted-foreground outline-none focus:ring-1 focus:ring-primary text-foreground transition-all hover:bg-secondary"
            />
          </div>

          {/* 2. Filter Tanggal Spesifik (Date Picker) */}
          <div className="relative inline-flex items-center">
            <Calendar className="absolute left-2.5 pointer-events-none size-3 text-muted-foreground" />
            <input
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              className="h-8 rounded-lg border bg-secondary/70 py-1 pl-7.5 pr-2.5 text-[10px] font-semibold outline-none focus:ring-1 focus:ring-primary text-foreground transition-all hover:bg-secondary cursor-pointer"
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
            options={["UNACKNOWLEDGED", "ACK", "RESOLVED"]}
          />

          {/* 8. Tombol Reset Filter */}
          {isFiltered && (
            <Button
              size="sm"
              variant="ghost"
              onClick={handleResetFilters}
              className="h-8 text-[10px] text-muted-foreground hover:text-foreground gap-1 px-2.5 cursor-pointer"
              title="Reset semua filter ke kondisi awal"
            >
              <RotateCcw className="size-3" />
              Reset Filter
            </Button>
          )}
        </div>
      </div>

      <RecordTable rows={filtered} onSelectRecord={onSelectRecord} />
      {filtered.length === 0 && (
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
    <label className="relative inline-block">
      <span className="sr-only">{label}</span>
      <select
        value={value}
        onChange={(e) => setValue(e.target.value)}
        className="h-8 appearance-none rounded-lg border bg-secondary/70 py-1 pl-3 pr-7.5 text-[10px] font-semibold outline-none focus:ring-1 focus:ring-primary text-foreground transition-all hover:bg-secondary cursor-pointer"
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
      <ChevronDown className="pointer-events-none absolute top-2.5 right-2 size-3 text-muted-foreground" />
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
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[950px] text-left">
        <thead className="bg-secondary/70 text-[9px] uppercase text-muted-foreground">
          <tr>
            <th className="px-5 py-3 font-bold">Tanggal</th>
            <th className="px-5 py-3 font-bold">Waktu</th>
            <th className="px-5 py-3 font-bold">Abnormality ID</th>
            <th className="px-5 py-3 font-bold">Kamera</th>
            <th className="px-5 py-3 font-bold">Objek</th>
            <th className="px-5 py-3 font-bold">Abnormality Type</th>
            <th className="px-5 py-3 font-bold">AI Conf.</th>
            <th className="px-5 py-3 font-bold">Severity</th>
            <th className="px-5 py-3 font-bold">Status Lifecycle</th>
            <th className="px-5 py-3 font-bold text-right">Aksi</th>
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
                <td className="px-5 py-3.5 whitespace-nowrap">
                  <div className="flex items-center gap-1.5 font-mono text-[10px] font-semibold text-foreground">
                    <Calendar className="size-3 text-primary/70 shrink-0" />
                    <span>{dateInfo.displayDate}</span>
                  </div>
                </td>

                {/* Kolom Waktu (Time) */}
                <td className="px-5 py-3.5 font-mono text-muted-foreground whitespace-nowrap">
                  {r.time}
                  {compact && " WIB"}
                </td>

                {/* Kolom Abnormality ID dengan Deep Link */}
                <td className="px-5 py-3.5 font-mono font-bold text-primary whitespace-nowrap">
                  <span className="inline-flex items-center gap-1 group-hover:underline">
                    {r.id}
                    <ExternalLink className="size-2.5 opacity-0 group-hover:opacity-100 transition-opacity text-primary" />
                  </span>
                </td>

                {/* Kamera */}
                <td className="px-5 py-3.5 font-bold whitespace-nowrap">{r.camera}</td>

                {/* Objek */}
                <td className="px-5 py-3.5 whitespace-nowrap">{r.object}</td>

                {/* Tipe Cacat */}
                <td
                  className={`px-5 py-3.5 font-semibold whitespace-nowrap ${
                    r.severity === "CRITICAL"
                      ? "text-danger"
                      : r.severity === "HIGH"
                      ? "text-primary"
                      : r.severity === "MEDIUM"
                      ? "text-warning"
                      : ""
                  }`}
                >
                  {r.type}
                </td>

                {/* AI Confidence */}
                <td className="px-5 py-3.5 font-mono font-bold whitespace-nowrap">{r.confidence}</td>

                {/* Severity */}
                <td className="px-5 py-3.5 whitespace-nowrap">
                  <Badge text={r.severity} />
                </td>

                {/* Status Lifecycle */}
                <td className="px-5 py-3.5 whitespace-nowrap">
                  <StatusBadge text={r.status} />
                </td>

                {/* Tombol Aksi: Lihat Bukti */}
                <td className="px-5 py-3.5 whitespace-nowrap text-right">
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
    text === "RESOLVED"
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
    id: "ABN-20261012-00002",
    inspectionId: 16,
    camera: "CAM-01",
    title: "Plat Bending #00002",
    severity: "MEDIUM",
    object: "Plat (Slat #4)",
    date: "2026-10-12",
    displayDate: "12 Okt 2026",
    time: "10:10:44 WIB",
    area: "Conveyor Plate & Rib Overhead",
    confidence: "92.00%",
    statusLifecycle: "UNACKNOWLEDGED",
  },
  {
    id: "ABN-20261012-00002",
    inspectionId: 16,
    camera: "CAM-01",
    title: "Rib Bending #00002",
    severity: "MEDIUM",
    object: "Rib (Slat #4)",
    date: "2026-10-12",
    displayDate: "12 Okt 2026",
    time: "10:10:44 WIB",
    area: "Conveyor Plate & Rib Overhead",
    confidence: "95.00%",
    statusLifecycle: "UNACKNOWLEDGED",
  },
  {
    id: "ABN-20261012-00001",
    inspectionId: 15,
    camera: "CAM-01",
    title: "Rib Bending #00001",
    severity: "MEDIUM",
    object: "Rib (Slat #3)",
    date: "2026-10-12",
    displayDate: "12 Okt 2026",
    time: "10:10:43 WIB",
    area: "Conveyor Plate & Rib Overhead",
    confidence: "95.00%",
    statusLifecycle: "UNACKNOWLEDGED",
  },
  {
    id: "ABN-20261009-00014",
    inspectionId: 14,
    camera: "CAM-01",
    title: "Rib Bending #00014",
    severity: "MEDIUM",
    object: "Rib (Slat #53)",
    date: "2026-10-09",
    displayDate: "09 Okt 2026",
    time: "21:15:52 WIB",
    area: "Conveyor Plate & Rib Overhead",
    confidence: "95.00%",
    statusLifecycle: "UNACKNOWLEDGED",
  },
  {
    id: "ABN-20261009-00013",
    inspectionId: 13,
    camera: "CAM-01",
    title: "Rib Bending #00013",
    severity: "MEDIUM",
    object: "Rib (Slat #52)",
    date: "2026-10-09",
    displayDate: "09 Okt 2026",
    time: "21:15:50 WIB",
    area: "Conveyor Plate & Rib Overhead",
    confidence: "95.00%",
    statusLifecycle: "UNACKNOWLEDGED",
  },
  {
    id: "ABN-20260913-00020",
    inspectionId: 5,
    camera: "CAM-02",
    title: "Bolt Loose #00020",
    severity: "HIGH",
    object: "Fastener M16",
    date: "2026-09-13",
    displayDate: "13 Sep 2026",
    time: "10:28:44 WIB",
    area: "Bolt & Pin Right Side",
    confidence: "97.20%",
    statusLifecycle: "ACKNOWLEDGED",
  },
  {
    id: "ABN-20260913-00019",
    inspectionId: 4,
    camera: "CAM-03",
    title: "Pin Missing #00019",
    severity: "MEDIUM",
    object: "Connecting Pin #11",
    date: "2026-09-13",
    displayDate: "13 Sep 2026",
    time: "10:21:05 WIB",
    area: "Bolt & Pin Left Side",
    confidence: "91.40%",
    statusLifecycle: "RESOLVED",
  },
];

function Evidence({
  evidenceList,
  onRefresh,
  initialSelectedId,
  onBackToHistory,
}: {
  evidenceList: EvidenceItem[];
  onRefresh?: () => void;
  initialSelectedId?: string | null;
  onBackToHistory?: () => void;
}) {
  const currentItems = evidenceList.length > 0 ? evidenceList : evidence;
  const [selected, setSelected] = useState<EvidenceItem>(() => {
    if (initialSelectedId) {
      const match = currentItems.find(
        (e) => e.id === initialSelectedId || e.inspectionId.toString() === initialSelectedId
      );
      if (match) return match;
    }
    return currentItems[0] || evidence[0]!;
  });
  const [open, setOpen] = useState<Record<string, boolean>>({
    "CAM-01": true,
    "CAM-02": false,
    "CAM-03": false,
  });
  const [openDates, setOpenDates] = useState<Record<string, boolean>>({});
  const [action, setAction] = useState(selected?.statusLifecycle || "UNACKNOWLEDGED");
  const [maintenanceAction, setMaintenanceAction] = useState(selected?.maintenanceNote || "");
  const [picRemark, setPicRemark] = useState(selected?.picName || "");

  useEffect(() => {
    if (selected?.camera) {
      setOpen((prev) => ({ ...prev, [selected.camera]: true }));
      const matchDate = extractEvidenceDate(selected).date;
      setOpenDates((prev) => ({
        ...prev,
        [`${selected.camera}_${matchDate}`]: true,
      }));
    }
  }, [selected?.camera]);

  // Ketika initialSelectedId dipilih dari Abnormality History, sinkronkan selected item & buka folder kamera serta subfolder tanggal
  useEffect(() => {
    if (!initialSelectedId) return;
    const match = currentItems.find(
      (e) => e.id === initialSelectedId || e.inspectionId.toString() === initialSelectedId
    );
    if (match) {
      setSelected(match);
      setAction(match.statusLifecycle || "UNACKNOWLEDGED");
      setMaintenanceAction(match.maintenanceNote || "");
      setPicRemark(match.picName || "");
      if (match.camera) {
        setOpen((prev) => ({ ...prev, [match.camera]: true }));
        const matchDate = extractEvidenceDate(match).date;
        setOpenDates((prev) => ({
          ...prev,
          [`${match.camera}_${matchDate}`]: true,
        }));
      }
    }
  }, [initialSelectedId, currentItems]);

  const toggleDateFolder = (key: string, currentOpenState: boolean) => {
    setOpenDates((prev) => ({
      ...prev,
      [key]: !currentOpenState,
    }));
  };

  const handleSelectItem = (item: EvidenceItem) => {
    setSelected(item);
    setAction(item.statusLifecycle || "UNACKNOWLEDGED");
    setMaintenanceAction(item.maintenanceNote || "");
    setPicRemark(item.picName || "");
  };

  // Sinkronisasi data background secara aman tanpa menimpa teks yang sedang diketik pengguna
  useEffect(() => {
    if (evidenceList.length === 0) return;

    const match = evidenceList.find((e) => e.id === selected?.id);
    if (match) {
      // Pertahankan teks form lokal yang sedang diketik teknisi (draft state)
      // Cukup sinkronkan metadata background (image, lifecycle dari DB)
      setSelected((prev) => ({
        ...match,
        statusLifecycle: action !== "UNACKNOWLEDGED" ? action : match.statusLifecycle,
      }));
    } else if (!selected || selected.inspectionId === 0) {
      // Hanya inisialisasi jika sebelumnya masih memakai mock data (inspectionId: 0)
      const first = evidenceList[0];
      setSelected(first);
      setAction(first.statusLifecycle || "UNACKNOWLEDGED");
      setMaintenanceAction(first.maintenanceNote || "");
      setPicRemark(first.picName || "");
    }
  }, [evidenceList]);

  if (!selected) return null;

  const userName = localStorage.getItem("userName") || "Sarah Jenkins";

  const handleAcknowledge = async () => {
    setAction("ACKNOWLEDGED");
    try {
      const picLabel = picRemark.trim() ? `${userName} - ${picRemark.trim()}` : userName;
      const noteLabel = maintenanceAction.trim() || "Dikonfirmasi oleh QA PIC";
      await submitInspectionAction(
        selected.inspectionId,
        "ACKNOWLEDGED",
        picLabel,
        noteLabel
      );
      if (onRefresh) onRefresh();
    } catch (e) {
      console.error(e);
    }
  };

  const handleResolve = async () => {
    setAction("RESOLVED");
    try {
      const picLabel = picRemark.trim() ? `${userName} - ${picRemark.trim()}` : userName;
      const noteLabel = maintenanceAction.trim() || "Telah diperbaiki & dikalibrasi ulang";
      await submitInspectionAction(
        selected.inspectionId,
        "RESOLVED",
        picLabel,
        noteLabel
      );
      if (onRefresh) onRefresh();
    } catch (e) {
      console.error(e);
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

  const imgUrl = getImageUrl(
    selected.annotatedPath || selected.rawImagePath || selected.imageUrl
  );

  return (
    <div className="space-y-5">
      <SectionTitle
        title="Evidence Archive & Area Inspection Explorer"
        subtitle="Jelajahi rekaman anomali tersimpan per area target kamera sesuai spesifikasi PRD"
        aside={
          <div className="flex items-center gap-2">
            {initialSelectedId && onBackToHistory && (
              <Button
                variant="outline"
                size="sm"
                onClick={onBackToHistory}
                className="h-7 text-[10px] gap-1 font-semibold text-primary border-primary/30 hover:bg-primary/10 cursor-pointer"
              >
                <ChevronLeft className="size-3" />
                Kembali ke Histori
              </Button>
            )}
            <StatusPill text="3 Target Camera Directories" />
            <StatusPill text="Sync Storage" />
          </div>
        }
      />

      {initialSelectedId && (
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
      <div className="grid gap-5 xl:grid-cols-[320px_1fr] items-start">
        <Panel className="overflow-hidden xl:sticky xl:top-4 self-start">
          <div className="border-b p-4">
            <h3 className="text-xs font-extrabold">Camera Area Explorer</h3>
            <p className="text-[9px] text-muted-foreground">
              Klasifikasi Rekaman Anomali • {currentItems.length} Files
            </p>
          </div>
          <div className="p-3 space-y-3">
            {cameraFolders.map((folder) => {
              const cam = folder.cam;
              const isOpen = open[cam];
              const items = currentItems.filter((e) => e.camera === cam);
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
                            Belum ada rekaman file pada folder ini
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
                                      const isSelected = selected.id === e.id && (selected.object === e.object || selected.inspectionId === e.inspectionId);

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
                                              className={`text-[9.5px] ${
                                                isSelected ? "text-primary font-extrabold" : "font-bold text-foreground"
                                              }`}
                                            >
                                              {normalizedTitle}
                                            </span>
                                            <Badge text={e.severity} />
                                          </div>
                                          <div className="mt-1 flex justify-between text-[8px] text-muted-foreground">
                                            <span className="truncate max-w-[140px]">{e.object}</span>
                                            <span className="font-mono shrink-0">{e.time}</span>
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
            <div className="camera-grid relative flex aspect-video max-h-[410px] items-center justify-center overflow-hidden bg-black/90">
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
            <div className="flex items-center justify-between">
              <div>
                <p className="text-[9px] font-bold uppercase text-muted-foreground">
                  Traceability ID
                </p>
                <h2 className="mt-1 font-mono text-base font-extrabold text-primary">
                  {selected.id}
                </h2>
              </div>
              <StatusBadge text={action} />
            </div>
            <dl className="mt-4 grid gap-4 border-y py-4 text-[10px] sm:grid-cols-2 lg:grid-cols-3">
              {[
                ["Target Camera", `${selected.camera} (${selected.area})`],
                ["Target Area", selected.area],
                ["Object", selected.object],
                ["Abnormality Type", selected.title.split(" #")[0] ?? ""],
                ["AI Confidence", selected.confidence],
                ["Capture Time", selected.captureTime || selected.time],
              ].map(([k, v]) => (
                <div key={k}>
                  <dt className="text-muted-foreground">{k}</dt>
                  <dd className="mt-1 font-bold">{v}</dd>
                </div>
              ))}
            </dl>
            <div className="mt-4 grid gap-4 sm:grid-cols-2">
              <label className="block text-[10px] font-bold">
                Tindakan Maintenance
                <textarea
                  value={maintenanceAction}
                  onChange={(e) => setMaintenanceAction(e.target.value)}
                  className="mt-2 min-h-20 w-full resize-none rounded-lg border bg-secondary p-3 text-[10px] outline-none focus:ring-1 focus:ring-ring text-foreground"
                  placeholder="Deskripsi tindakan perbaikan atau maintenance yang dilakukan..."
                />
              </label>
              <label className="block text-[10px] font-bold">
                PIC Remark
                <textarea
                  value={picRemark}
                  onChange={(e) => setPicRemark(e.target.value)}
                  className="mt-2 min-h-20 w-full resize-none rounded-lg border bg-secondary p-3 text-[10px] outline-none focus:ring-1 focus:ring-ring text-foreground"
                  placeholder="Catatan verifikasi PIC / remark teknisi..."
                />
              </label>
            </div>
            <div className="mt-3 flex gap-2">
              <Button
                variant="secondary"
                size="sm"
                onClick={handleAcknowledge}
                disabled={action === "ACKNOWLEDGED" || action === "RESOLVED"}
              >
                <Wrench className="size-4" />
                Acknowledge
              </Button>
              <Button
                size="sm"
                className="bg-success text-white hover:bg-success/90"
                onClick={handleResolve}
                disabled={action === "RESOLVED"}
              >
                <Check className="size-4" />
                Mark Resolved
              </Button>
            </div>
          </Panel>
        </div>
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

  return (
    <div className="space-y-5">
      <SectionTitle
        title="Analytics & Maintenance Insights"
        subtitle="Pola cacat berulang (recurring abnormality) dan performa inspeksi berdasarkan PRD Section 20"
      />
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
