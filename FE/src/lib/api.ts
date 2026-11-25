const API_HOST = typeof window !== "undefined" ? window.location.hostname : "localhost";
export const API_BASE_URL = `http://${API_HOST}:5000/api`;

export const SCALE_FACTOR_MM_PER_PX = 0.515;

/**
 * Format nilai piksel dan milimeter secara seragam: mm di depan, px di dalam kurung (1 desimal).
 * Contoh: 14.0 px -> "11.5 mm (14.0 px)"
 */
export function formatPxAndMm(pxVal?: number | null, scale = SCALE_FACTOR_MM_PER_PX): string {
  if (pxVal === undefined || pxVal === null || isNaN(pxVal)) return "-";
  const mmVal = pxVal * scale;
  return `${mmVal.toFixed(1)} mm (${pxVal.toFixed(1)} px)`;
}

export type Severity = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "NONE";

export type AbnormalComponentSummary = {
  id: string;
  label: string;
  shortLabel: string;
  camera: string;
  count: number;
  activeCount?: number;
  resolvedCount?: number;
  dotColor: string;
};

export type RecordRow = {
  id: string;
  inspectionId: number;
  detectionId?: number;
  date?: string;
  displayDate?: string;
  time: string;
  fullTime?: string;
  camera: string;
  object: string;
  type: string;
  confidence: string;
  severity: Severity;
  status: string;
  statusLifecycle: string;
  overallResult?: string;
  detectionResult?: string;
  resolvedAt?: string | null;
  resolvedAtDisplay?: string | null;
  picName?: string;
  maintenanceNote?: string;
  pvFlatness?: number | null;
  maxDeviation?: number | null;
  pvFlatnessMm?: number | null;
  maxDeviationMm?: number | null;
  rawImagePath?: string | null;
  annotatedPath?: string | null;
};

export const MONTH_NAMES_ID = [
  "Januari",
  "Februari",
  "Maret",
  "April",
  "Mei",
  "Juni",
  "Juli",
  "Agustus",
  "September",
  "Oktober",
  "November",
  "Desember",
] as const;

export const MONTH_SHORT_ID = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "Mei",
  "Jun",
  "Jul",
  "Ags",
  "Sep",
  "Okt",
  "Nov",
  "Des",
] as const;

/**
 * Ekstraksi informasi penanggalan (tanggal, bulan, tahun) dari RecordRow
 */
export function extractRecordDate(r: RecordRow): {
  date: string; // YYYY-MM-DD
  displayDate: string; // misal: "12 Okt 2026"
  year: number;
  month: number; // 1-12
  day: number;
} {
  if (r.date && r.date.includes("-")) {
    const parts = r.date.split("-");
    const y = parseInt(parts[0], 10);
    const m = parseInt(parts[1], 10);
    const d = parseInt(parts[2], 10);
    const display = r.displayDate || `${d} ${MONTH_SHORT_ID[m - 1] || m} ${y}`;
    return { date: r.date, displayDate: display, year: y, month: m, day: d };
  }

  // Fallback: Ekstraksi dari pola ID ABN-YYYYMMDD-XXXXX
  const idMatch = r.id?.match(/ABN-(\d{4})(\d{2})(\d{2})-/i);
  if (idMatch) {
    const y = parseInt(idMatch[1], 10);
    const m = parseInt(idMatch[2], 10);
    const d = parseInt(idMatch[3], 10);
    const dateFormatted = `${y}-${String(m).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
    const display = `${d} ${MONTH_SHORT_ID[m - 1] || m} ${y}`;
    return { date: dateFormatted, displayDate: display, year: y, month: m, day: d };
  }

  // Fallback default: Tanggal hari ini
  const now = new Date();
  const y = now.getFullYear();
  const m = now.getMonth() + 1;
  const d = now.getDate();
  const dateFormatted = `${y}-${String(m).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
  return {
    date: dateFormatted,
    displayDate: `${d} ${MONTH_SHORT_ID[m - 1] || m} ${y}`,
    year: y,
    month: m,
    day: d,
  };
}

export type EvidenceItem = {
  id: string;
  inspectionId: number;
  camera: string;
  title: string;
  severity: Severity;
  object: string;
  date?: string;
  displayDate?: string;
  time: string;
  captureTime?: string;
  area: string;
  confidence: string;
  statusLifecycle?: string;
  overallResult?: string;
  detectionResult?: string;
  resolvedAt?: string | null;
  resolvedAtDisplay?: string | null;
  picName?: string;
  maintenanceNote?: string;
  pvFlatness?: number | null;
  maxDeviation?: number | null;
  pvFlatnessMm?: number | null;
  maxDeviationMm?: number | null;
  rawImagePath?: string | null;
  annotatedPath?: string | null;
  imageUrl?: string | null;
};

/**
 * Ekstraksi informasi penanggalan dari EvidenceItem
 */
export function extractEvidenceDate(e: EvidenceItem): {
  date: string; // YYYY-MM-DD
  displayDate: string; // misal: "12 Okt 2026"
  year: number;
  month: number;
  day: number;
} {
  if (e.date && e.date.includes("-")) {
    const parts = e.date.split("-");
    const y = parseInt(parts[0], 10);
    const m = parseInt(parts[1], 10);
    const d = parseInt(parts[2], 10);
    const display = e.displayDate || `${d} ${MONTH_SHORT_ID[m - 1] || m} ${y}`;
    return { date: e.date, displayDate: display, year: y, month: m, day: d };
  }

  // Fallback 1: Pola ID ABN-YYYYMMDD-XXXXX
  const idMatch = e.id?.match(/ABN-(\d{4})(\d{2})(\d{2})-/i);
  if (idMatch) {
    const y = parseInt(idMatch[1], 10);
    const m = parseInt(idMatch[2], 10);
    const d = parseInt(idMatch[3], 10);
    const dateFormatted = `${y}-${String(m).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
    const display = `${d} ${MONTH_SHORT_ID[m - 1] || m} ${y}`;
    return { date: dateFormatted, displayDate: display, year: y, month: m, day: d };
  }

  // Fallback 2: captureTime string (misal "12 Oct 2026, 20:10:44")
  if (e.captureTime) {
    const parsed = new Date(e.captureTime.replace(" WIB", ""));
    if (!isNaN(parsed.getTime())) {
      const y = parsed.getFullYear();
      const m = parsed.getMonth() + 1;
      const d = parsed.getDate();
      const dateFormatted = `${y}-${String(m).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
      const display = `${d} ${MONTH_SHORT_ID[m - 1] || m} ${y}`;
      return { date: dateFormatted, displayDate: display, year: y, month: m, day: d };
    }
  }

  // Fallback default: Hari ini
  const now = new Date();
  const y = now.getFullYear();
  const m = now.getMonth() + 1;
  const d = now.getDate();
  const dateFormatted = `${y}-${String(m).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
  return {
    date: dateFormatted,
    displayDate: `${d} ${MONTH_SHORT_ID[m - 1] || m} ${y}`,
    year: y,
    month: m,
    day: d,
  };
}

export type MaintenanceLogItem = {
  logId: number;
  inspectionId: number;
  action: "ACKNOWLEDGED" | "RESOLVED" | string;
  picName: string;
  notes: string;
  isFalseAlarm?: boolean;
  isReopen?: boolean;
  loggedAt: string;
  loggedAtDisplay: string;
};

export type AnalyticsData = {
  weekly: { d: string; n: number }[];
  pareto: { label: string; n: number; p: string; barCls: string }[];
  totalIncidents: number;
  resolutionSummary?: {
    openCount: number;
    resolvedCount: number;
    invalidCount: number;
    resolutionRatePct: number;
    avgMttrMinutes: number | null;
  };
};

/**
 * Mendapatkan URL gambar dari backend Flask image server.
 */
export function getImageUrl(path?: string | null): string | null {
  if (!path) return null;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  if (path.startsWith("/api/image")) {
    return `http://${API_HOST}:5000${path}`;
  }
  return `${API_BASE_URL}/image?path=${encodeURIComponent(path)}`;
}

/**
 * Cek status kesehatan backend API & database.
 */
export async function fetchHealth(): Promise<{ status: string; database: string }> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) throw new Error("Health check failed");
    return await res.json();
  } catch (err) {
    return { status: "offline", database: "disconnected" };
  }
}

/**
 * Mengambil ringkasan 6 kartu statistik abnormalitas.
 */
export async function fetchStats(): Promise<{
  components: AbnormalComponentSummary[];
  overview: {
    totalInspections: number;
    totalNG: number;
    activeNG?: number;
    resolvedNG?: number;
    invalidCount?: number;
    totalOK: number;
    accuracy: string;
  };
}> {
  const res = await fetch(`${API_BASE_URL}/stats`);
  if (!res.ok) throw new Error("Gagal mengambil data statistik");
  return await res.json();
}

/**
 * Mengambil histori anomali dengan filter.
 */
export async function fetchInspections(params?: {
  camera?: string;
  severity?: string;
  status?: string;
  limit?: number;
  days?: number;
}): Promise<RecordRow[]> {
  const query = new URLSearchParams();
  if (params?.camera && params.camera !== "ALL") query.append("camera", params.camera);
  if (params?.severity && params.severity !== "ALL") query.append("severity", params.severity);
  if (params?.status && params.status !== "ALL") query.append("status", params.status);
  const limit = params?.limit ?? 300;
  query.append("limit", limit.toString());
  const days = params?.days ?? 7;
  query.append("days", days.toString());

  const res = await fetch(`${API_BASE_URL}/inspections?${query.toString()}`);
  if (!res.ok) throw new Error("Gagal mengambil data inspeksi");
  return await res.json();
}

/**
 * Mengambil daftar bukti visual untuk Evidence Viewer.
 */
export async function fetchEvidence(): Promise<EvidenceItem[]> {
  const res = await fetch(`${API_BASE_URL}/evidence`);
  if (!res.ok) throw new Error("Gagal mengambil data bukti visual");
  return await res.json();
}

/**
 * Mengirim aksi teknisi / PIC (Acknowledge, Mark Resolved, False Alarm, atau Reopen).
 */
export async function submitInspectionAction(
  inspectionId: number,
  action: "ACKNOWLEDGED" | "RESOLVED" | "FALSE_ALARM" | "UNACKNOWLEDGED",
  picName: string,
  maintenanceNote: string
): Promise<{
  success: boolean;
  inspectionId: number;
  statusLifecycle: string;
  overallResult?: string;
  resolvedAt?: string | null;
  resolvedAtDisplay?: string | null;
  picName?: string;
  maintenanceNote?: string;
}> {
  const res = await fetch(`${API_BASE_URL}/inspections/${inspectionId}/action`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      action,
      picName,
      maintenanceNote,
    }),
  });

  if (!res.ok) {
    let errMsg = `Gagal memperbarui status inspeksi (${res.status})`;
    try {
      const errBody = await res.json();
      if (errBody?.error) errMsg = errBody.error;
    } catch {
      // ignore json parse failure
    }
    throw new Error(errMsg);
  }
  return await res.json();
}

/**
 * Mengambil riwayat audit log tindakan PIC untuk suatu inspeksi.
 */
export async function fetchInspectionLogs(inspectionId: number): Promise<MaintenanceLogItem[]> {
  const res = await fetch(`${API_BASE_URL}/inspections/${inspectionId}/logs`);
  if (!res.ok) return [];
  return await res.json();
}

/**
 * Mengambil data Analytics (Weekly trend & Pareto).
 */
export async function fetchAnalytics(): Promise<AnalyticsData> {
  const res = await fetch(`${API_BASE_URL}/analytics`);
  if (!res.ok) throw new Error("Gagal mengambil data analitik");
  return await res.json();
}

export interface TopInspectionThresholds {
  rib_tip_threshold: number;
  rib_dev_threshold: number;
  rib_max_dev_threshold?: number;
  rib_pv_threshold: number;
  rib_sagitta_threshold: number;
  rib_conf_threshold: number;
  max_flatness_pv: number;
  max_sagitta: number;
  max_center_void: number;
  min_continuity_cov?: number;
  slat_max_gap_limit?: number;
  slat_delta_limit?: number;
  slat_valley_sens?: number;
  scale_factor_mm_per_px?: number;
  [key: string]: any;
}

export interface TopInspectionConfig {
  camera_id: string;
  thresholds: TopInspectionThresholds;
  updated_at?: string;
  [key: string]: any;
}

/**
 * Mengambil konfigurasi tuning top inspection terkini (Hot-Reload).
 */
export async function fetchTopInspectionConfig(): Promise<{ status: string; config: TopInspectionConfig }> {
  const res = await fetch(`${API_BASE_URL}/config/top-inspection`);
  if (!res.ok) throw new Error("Gagal mengambil konfigurasi inspeksi");
  return await res.json();
}

/**
 * Memperbarui parameter tuning top inspection (memicu Hot-Reload di OpenCV tanpa restart).
 */
export async function updateTopInspectionConfig(
  thresholds: Partial<TopInspectionThresholds>
): Promise<{ status: string; message: string; thresholds: TopInspectionThresholds; updated_at: string }> {
  const res = await fetch(`${API_BASE_URL}/config/top-inspection`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ thresholds }),
  });
  if (!res.ok) throw new Error("Gagal memperbarui konfigurasi inspeksi");
  return await res.json();
}

