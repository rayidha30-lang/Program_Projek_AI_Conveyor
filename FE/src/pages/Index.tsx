import React, { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  AlertTriangle,
  BarChart3,
  Camera,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Cpu,
  Download,
  Eye,
  Folder,
  FolderOpen,
  Gauge,
  Layers,
  LayoutDashboard,
  LogOut,
  ShieldAlert,
  ShieldCheck,
  Wrench,
} from "lucide-react";
import { Button } from "@/components/ui/button";

type View = "dashboard" | "live" | "abnormality" | "evidence" | "analytics";
type Severity = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
type RecordRow = {
  time: string;
  id: string;
  camera: string;
  object: string;
  type: string;
  confidence: string;
  severity: Severity;
  status: string;
};

type AbnormalComponentSummary = {
  id: string;
  label: string;
  shortLabel: string;
  camera: string;
  count: number;
  dotColor: string;
};

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
    time: "10:31:22",
    id: "ABN-20260913-00021",
    camera: "CAM-01",
    object: "Rib Plate #42",
    type: "Rib Deformation",
    confidence: "94.8%",
    severity: "CRITICAL",
    status: "UNACKNOWLEDGED",
  },
  {
    time: "10:28:44",
    id: "ABN-20260913-00020",
    camera: "CAM-02",
    object: "Fastener M16",
    type: "Bolt Missing",
    confidence: "97.2%",
    severity: "HIGH",
    status: "ACK: Rudi S.",
  },
  {
    time: "10:21:05",
    id: "ABN-20260913-00019",
    camera: "CAM-03",
    object: "Connecting Pin",
    type: "Pin Missing",
    confidence: "91.4%",
    severity: "MEDIUM",
    status: "RESOLVED",
  },
  {
    time: "10:14:18",
    id: "ABN-20260913-00018",
    camera: "CAM-01",
    object: "Slat C-09",
    type: "Plate Deformation",
    confidence: "88.0%",
    severity: "LOW",
    status: "RESOLVED",
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
  const [isCollapsed, setIsCollapsed] = useState(false);

  return (
    <div className="min-h-screen bg-background text-foreground lg:h-screen lg:overflow-hidden">
      <Sidebar
        view={view}
        setView={setView}
        isCollapsed={isCollapsed}
        setIsCollapsed={setIsCollapsed}
      />
      <div
        className={`transition-all duration-300 ease-in-out lg:flex lg:h-screen lg:flex-col ${
          isCollapsed ? "lg:ml-24" : "lg:ml-72"
        }`}
      >
        <Header view={view} />
        <main className="flex-1 overflow-y-auto px-4 py-5 sm:px-6 lg:px-8 lg:py-7">
          {view === "dashboard" && (
            <Dashboard onViewAll={() => setView("abnormality")} />
          )}
          {view === "live" && <LiveCamera />}
          {view === "abnormality" && <Abnormality />}
          {view === "evidence" && <Evidence />}
          {view === "analytics" && <Analytics />}
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
}: {
  view: View;
  setView: (view: View) => void;
  isCollapsed: boolean;
  setIsCollapsed: React.Dispatch<React.SetStateAction<boolean>>;
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

  return (
    <aside
      className={`fixed inset-y-3 left-3 z-30 flex flex-col justify-between rounded-3xl border border-white/20 bg-linear-to-b from-[#7645f7] via-[#6a37eb] to-[#5a24e2] text-white shadow-2xl shadow-purple-900/30 transition-all duration-300 ease-in-out ${
        isCollapsed ? "w-20 py-5 px-2 items-center" : "w-64 py-5 px-4"
      }`}
    >
      <div>
        {isCollapsed ? (
          <div className="flex flex-col items-center gap-3">
            {/* Logo Cloud / Vision icon */}
            <button
              onClick={() => setIsCollapsed(false)}
              className="flex size-11 items-center justify-center rounded-2xl bg-white/20 text-white shadow-md hover:bg-white/30 transition-all cursor-pointer"
              title="Perbesar Side Bar"
            >
              <Cpu className="size-6" />
            </button>
            <button
              onClick={() => setIsCollapsed(false)}
              className="flex size-7 items-center justify-center rounded-lg bg-white/10 text-white/80 hover:bg-white/25 hover:text-white transition-colors cursor-pointer"
              title="Perbesar Side Bar"
            >
              <ChevronRight className="size-4" />
            </button>
          </div>
        ) : (
          <div className="mb-6 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex size-10 items-center justify-center rounded-2xl bg-white/20 shadow-md text-white">
                <Cpu className="size-5" />
              </div>
              <div className="overflow-hidden">
                <p className="text-xs font-extrabold text-white leading-tight truncate">
                  Die Cast Machine #2
                </p>
                <p className="mt-0.5 text-[8.5px] font-bold tracking-wider text-purple-200 truncate">
                  SLAT RIB VISION SYSTEM
                </p>
              </div>
            </div>
            <button
              onClick={() => setIsCollapsed(true)}
              className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-white/10 text-white/80 hover:bg-white/20 hover:text-white transition-colors cursor-pointer"
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
          {nav.map((item) => {
            const Icon = item.icon;
            const active = view === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setView(item.id)}
                title={item.label}
                className={`group relative flex items-center transition-all duration-200 cursor-pointer ${
                  isCollapsed
                    ? `size-11 justify-center rounded-2xl ${
                        active
                          ? "bg-black/25 text-white shadow-inner ring-1 ring-white/30"
                          : "text-white/80 hover:bg-white/15 hover:text-white"
                      }`
                    : `h-11 w-full gap-3 rounded-xl px-3.5 text-xs font-bold ${
                        active
                          ? "bg-black/20 text-white shadow-inner ring-1 ring-white/30"
                          : "text-white/80 hover:bg-white/15 hover:text-white"
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
          {/* User Avatar */}
          <div
            className="group relative flex size-10 items-center justify-center overflow-hidden rounded-xl border-2 border-white/50 bg-white/20 shadow-md cursor-pointer"
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
            className="flex size-10 items-center justify-center rounded-xl text-white/80 hover:bg-white/20 hover:text-white transition-colors cursor-pointer"
            title="Logout System"
          >
            <LogOut className="size-4.5" />
          </button>
        </div>
      ) : (
        <div className="space-y-3 pt-4">
          <div className="rounded-xl border border-white/20 bg-white/10 p-3 backdrop-blur-xs text-white">
            <div className="mb-1.5 flex items-center justify-between text-[10px]">
              <span className="text-purple-200">System Status</span>
              <span className="flex items-center gap-1 font-bold text-white">
                <span className="size-1.5 rounded-full bg-emerald-400 animate-pulse" />{" "}
                ONLINE
              </span>
            </div>
            <div className="flex justify-between font-mono text-[9px] text-purple-200">
              <span>Uptime: 99.8%</span>
              <span>SSMS Connected</span>
            </div>
          </div>

          <div className="flex items-center justify-between pt-1">
            <div className="flex items-center gap-2.5 overflow-hidden">
              <div className="size-9 shrink-0 overflow-hidden rounded-xl border-2 border-white/50 bg-white/20 shadow-sm relative flex items-center justify-center text-xs font-bold">
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
                <p className="text-[9px] text-purple-200 truncate">
                  QA Lead / PIC
                </p>
              </div>
            </div>
            <button
              onClick={handleLogout}
              className="flex size-8 shrink-0 items-center justify-center rounded-lg text-white/80 hover:bg-white/20 hover:text-white transition-colors cursor-pointer"
              title="Logout System"
            >
              <LogOut className="size-4" />
            </button>
          </div>
        </div>
      )}
    </aside>
  );
}

function Header({ view }: { view: View }) {
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
          <StatusPill text="API: Online" />
          <StatusPill text="3 Cams: Online" />
          <span className="flex items-center gap-1.5 rounded-lg border border-border/60 bg-card px-2.5 py-1.5 font-mono text-[9px] font-semibold text-muted-foreground shadow-xs">
            <Clock3 className="size-3" />
            13 Sep 2026 | 11:44 WIB
          </span>
        </div>
      </div>
    </header>
  );
}

function StatusPill({ text }: { text: string }) {
  return (
    <span className="flex items-center gap-1 rounded-full border border-success/30 bg-success-soft px-2 py-1 text-[9px] font-bold text-success">
      <span className="size-1.5 rounded-full bg-success" />
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

function Dashboard({ onViewAll }: { onViewAll: () => void }) {
  const stats = [
    {
      label: "Total Abnormal Plat/Slat",
      value: "5",
      camera: "CAM-01 (Overhead)",
      note: "Plate Deform / Crack",
      Icon: Layers,
      tone: "text-danger",
      bgTone: "bg-danger-soft text-danger",
      dotColor: "bg-danger",
    },
    {
      label: "Total Abnormal Rib/Sirip",
      value: "3",
      camera: "CAM-01 (Overhead)",
      note: "Rib Bending / Profile",
      Icon: AlertTriangle,
      tone: "text-warning",
      bgTone: "bg-warning-soft text-warning",
      dotColor: "bg-warning",
    },
    {
      label: "Total Abnormal Bolt Sisi Kanan",
      value: "4",
      camera: "CAM-02 (RH Side)",
      note: "Loose / Missing Bolt",
      Icon: Wrench,
      tone: "text-info",
      bgTone: "bg-info-soft text-info",
      dotColor: "bg-info",
    },
    {
      label: "Total Abnormal Pin Sisi Kanan",
      value: "2",
      camera: "CAM-02 (RH Side)",
      note: "Dislodged Link Pin",
      Icon: ShieldAlert,
      tone: "text-primary",
      bgTone: "bg-primary/10 text-primary",
      dotColor: "bg-primary",
    },
    {
      label: "Total Abnormal Bolt Sisi Kiri",
      value: "3",
      camera: "CAM-03 (LH Side)",
      note: "Loose / Untorqued",
      Icon: Wrench,
      tone: "text-primary",
      bgTone: "bg-primary/10 text-primary",
      dotColor: "bg-primary",
    },
    {
      label: "Total Abnormal Pin Sisi Kiri",
      value: "1",
      camera: "CAM-03 (LH Side)",
      note: "Critical Missing Pin",
      Icon: AlertTriangle,
      tone: "text-danger",
      bgTone: "bg-danger-soft text-danger",
      dotColor: "bg-danger",
    },
  ];

  return (
    <div className="space-y-6">
      <div className="grid gap-3.5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
        {stats.map((stat) => {
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
        <RecordTable rows={records.slice(0, 3)} compact />
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

function Abnormality() {
  const [camera, setCamera] = useState("ALL");
  const [severity, setSeverity] = useState("ALL");
  const [status, setStatus] = useState("ALL");

  const filtered = useMemo(
    () =>
      records.filter(
        (r) =>
          (camera === "ALL" || r.camera === camera) &&
          (severity === "ALL" || r.severity === severity) &&
          (status === "ALL" || r.status.startsWith(status))
      ),
    [camera, severity, status]
  );

  function exportCsv() {
    const head =
      "Time,Abnormality ID,Camera,Object,Type,Confidence,Severity,Status\n";
    const csv =
      head + filtered.map((r) => Object.values(r).join(",")).join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = "abnormality-history.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  return (
    <Panel className="overflow-hidden">
      <div className="p-5 pb-4">
        <SectionTitle
          title="Abnormality History & Traceability Ledger"
          subtitle="Pencatatan seluruh histori kelainan dengan kemampuan filter multi-parameter"
        />
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <Filter
            value={camera}
            setValue={setCamera}
            label="Semua Kamera (CAM 01, 02, 03)"
            options={["CAM-01", "CAM-02", "CAM-03"]}
          />
          <Filter
            value={severity}
            setValue={setSeverity}
            label="Semua Severity"
            options={["CRITICAL", "HIGH", "MEDIUM", "LOW"]}
          />
          <Filter
            value={status}
            setValue={setStatus}
            label="Semua Status"
            options={["UNACKNOWLEDGED", "ACK", "RESOLVED"]}
          />
          <Button size="sm" onClick={exportCsv} className="h-8 text-xs">
            <Download className="size-3.5" />
            Export CSV
          </Button>
        </div>
      </div>
      <RecordTable rows={filtered} />
      {filtered.length === 0 && (
        <p className="p-8 text-center text-xs text-muted-foreground">
          No records match the selected filters.
        </p>
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
  options: string[] | readonly string[];
}) {
  return (
    <label className="relative inline-block">
      <span className="sr-only">{label}</span>
      <select
        value={value}
        onChange={(e) => setValue(e.target.value)}
        className="h-8 appearance-none rounded-lg border bg-secondary py-1 pl-3 pr-8 text-[10px] font-semibold outline-none focus:ring-1 focus:ring-ring text-foreground"
      >
        <option value="ALL">{label}</option>
        {options.map((x) => (
          <option key={x} value={x}>
            {x}
          </option>
        ))}
      </select>
      <ChevronDown className="pointer-events-none absolute top-2.5 right-2 size-3 text-muted-foreground" />
    </label>
  );
}

function RecordTable({
  rows,
  compact = false,
}: {
  rows: RecordRow[];
  compact?: boolean;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[850px] text-left">
        <thead className="bg-secondary/70 text-[9px] uppercase text-muted-foreground">
          <tr>
            {[
              "Time",
              "Abnormality ID",
              "Camera",
              "Object",
              "Abnormality Type",
              "AI Conf.",
              "Severity",
              "Status Lifecycle",
            ].map((x) => (
              <th key={x} className="px-5 py-3 font-bold">
                {x}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {rows.map((r) => (
            <tr key={r.id} className="text-[10px] hover:bg-secondary/40">
              <td className="px-5 py-4 font-mono text-muted-foreground">
                {r.time}
                {compact && " WIB"}
              </td>
              <td className="px-5 py-4 font-mono font-bold text-primary">
                {r.id}
              </td>
              <td className="px-5 py-4 font-bold">{r.camera}</td>
              <td className="px-5 py-4">{r.object}</td>
              <td
                className={`px-5 py-4 font-semibold ${
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
              <td className="px-5 py-4 font-mono font-bold">{r.confidence}</td>
              <td className="px-5 py-4">
                <Badge text={r.severity} />
              </td>
              <td className="px-5 py-4">
                <StatusBadge text={r.status} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Badge({ text }: { text: Severity }) {
  const cls = {
    CRITICAL: "border-danger/30 bg-danger-soft text-danger",
    HIGH: "border-primary/30 bg-info-soft text-primary",
    MEDIUM: "border-warning/40 bg-warning-soft text-warning",
    LOW: "border-border bg-secondary text-muted-foreground",
  }[text];
  return (
    <span className={`rounded border px-2 py-0.5 text-[8px] font-bold ${cls}`}>
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

type EvidenceItem = {
  id: string;
  camera: string;
  title: string;
  severity: Severity;
  object: string;
  time: string;
  area: string;
  confidence: string;
};

const evidence: EvidenceItem[] = [
  {
    id: "ABN-20260913-00021",
    camera: "CAM-01",
    title: "Rib Deformation #00021",
    severity: "CRITICAL",
    object: "Rib Plate #42",
    time: "10:31:22 WIB",
    area: "Conveyor Plate & Rib Overhead",
    confidence: "94.82%",
  },
  {
    id: "ABN-20260913-00018",
    camera: "CAM-01",
    title: "Plate Crack #00018",
    severity: "LOW",
    object: "Slat Plate C-09",
    time: "10:14:18 WIB",
    area: "Conveyor Plate & Rib Overhead",
    confidence: "88.00%",
  },
  {
    id: "ABN-20260913-00014",
    camera: "CAM-01",
    title: "Plate Misalignment #00014",
    severity: "MEDIUM",
    object: "Slat Joint B-12",
    time: "09:48:02 WIB",
    area: "Conveyor Plate & Rib Overhead",
    confidence: "90.21%",
  },
  {
    id: "ABN-20260913-00020",
    camera: "CAM-02",
    title: "Bolt Loose / Untorqued #00020",
    severity: "HIGH",
    object: "Fastener M16",
    time: "10:28:44 WIB",
    area: "Bolt & Pin Right Side",
    confidence: "97.20%",
  },
  {
    id: "ABN-20260913-00015",
    camera: "CAM-02",
    title: "Bolt Missing #00015",
    severity: "CRITICAL",
    object: "RH Guide Bolt #18",
    time: "09:55:10 WIB",
    area: "Bolt & Pin Right Side",
    confidence: "96.40%",
  },
  {
    id: "ABN-20260913-00019",
    camera: "CAM-03",
    title: "Pin Missing / Dislodged #00019",
    severity: "MEDIUM",
    object: "Connecting Pin #11",
    time: "10:21:05 WIB",
    area: "Bolt & Pin Left Side",
    confidence: "91.40%",
  },
];

function Evidence() {
  const [selected, setSelected] = useState<EvidenceItem>(evidence[0]!);
  const [open, setOpen] = useState<Record<string, boolean>>({
    "CAM-01": true,
    "CAM-02": true,
    "CAM-03": true,
  });
  const [action, setAction] = useState("UNACKNOWLEDGED");
  const [maintenanceAction, setMaintenanceAction] = useState("");
  const [picRemark, setPicRemark] = useState("");

  if (!selected) return null;

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

  return (
    <div className="space-y-5">
      <SectionTitle
        title="Evidence Archive & Area Inspection Explorer"
        subtitle="Jelajahi rekaman anomali tersimpan per area target kamera sesuai spesifikasi PRD"
        aside={
          <div className="flex gap-2">
            <StatusPill text="3 Target Camera Directories" />
            <StatusPill text="Sync Storage" />
          </div>
        }
      />
      <div className="grid gap-5 xl:grid-cols-[320px_1fr]">
        <Panel className="overflow-hidden">
          <div className="border-b p-4">
            <h3 className="text-xs font-extrabold">Camera Area Explorer</h3>
            <p className="text-[9px] text-muted-foreground">
              Klasifikasi Rekaman Anomali • 6 Files
            </p>
          </div>
          <div className="p-3 space-y-3">
            {cameraFolders.map((folder) => {
              const cam = folder.cam;
              const isOpen = open[cam];
              const items = evidence.filter((e) => e.camera === cam);
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
                  {isOpen && (
                    <div className="space-y-1.5 pt-2 px-1 pb-1">
                      {items.map((e) => (
                        <button
                          key={e.id}
                          type="button"
                          onClick={() => {
                            setSelected(e);
                            setAction("UNACKNOWLEDGED");
                            setMaintenanceAction("");
                            setPicRemark("");
                          }}
                          className={`w-full rounded-xl border p-2.5 text-left transition-all cursor-pointer ${
                            selected.id === e.id
                              ? "border-primary bg-info-soft shadow-xs"
                              : "border-border/70 bg-card hover:bg-secondary/70"
                          }`}
                        >
                          <div className="flex justify-between items-center gap-2">
                            <span
                              className={`text-[10px] font-bold ${
                                selected.id === e.id ? "text-primary font-extrabold" : "text-foreground"
                              }`}
                            >
                              {e.title}
                            </span>
                            <Badge text={e.severity} />
                          </div>
                          <div className="mt-1 flex justify-between text-[8.5px] text-muted-foreground">
                            <span>{e.object}</span>
                            <span className="font-mono">{e.time}</span>
                          </div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </Panel>
        <div className="space-y-5">
          <Panel className="overflow-hidden">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b p-4">
              <div>
                <h3 className="text-xs font-extrabold">
                  RAW Evidence Optical Capture • {selected.camera}
                </h3>
                <p className="mt-1 font-mono text-[9px] text-muted-foreground">
                  {selected.camera.replace("-", "")}_20260913_
                  {selected.time.replaceAll(":", "").slice(0, 6)}_
                  {selected.id.slice(-8)}.jpg
                </p>
              </div>
              <Badge text={selected.severity} />
            </div>
            <div className="camera-grid relative flex aspect-video max-h-[410px] items-center justify-center overflow-hidden">
              <div className="h-2/5 w-1/3 border-2 border-danger">
                <span className="-mt-6 block w-fit bg-danger px-2 py-1 font-mono text-[9px] font-bold text-primary-foreground">
                  {selected.title.toUpperCase()} • {selected.confidence} CONF
                </span>
              </div>
              <span className="absolute bottom-4 left-4 rounded bg-camera/80 px-2 py-1 font-mono text-[8px] text-sidebar-foreground">
                SHA-256 VERIFIED
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
                ["Capture Time", `13 Sep 2026, ${selected.time}`],
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
                onClick={() => setAction("ACKNOWLEDGED")}
              >
                <Wrench className="size-4" />
                Acknowledge
              </Button>
              <Button
                size="sm"
                className="bg-success text-white hover:bg-success/90"
                onClick={() => setAction("RESOLVED")}
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

function Analytics() {
  const days = [
    { d: "Sen", n: 4 },
    { d: "Sel", n: 7 },
    { d: "Rab", n: 3 },
    { d: "Kam", n: 9 },
    { d: "Jum", n: 12 },
    { d: "Sab", n: 5 },
    { d: "Min", n: 2 },
  ];

  const pareto = [
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
                  style={{ height: `${(item.n / 12) * 75}%` }}
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
            subtitle="Total: 75 Insiden"
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
