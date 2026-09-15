export const API_BASE_URL = "http://localhost:5000/api";

export type Severity = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "NONE";

export type AbnormalComponentSummary = {
  id: string;
  label: string;
  shortLabel: string;
  camera: string;
  count: number;
  dotColor: string;
};

export type RecordRow = {
  id: string;
  inspectionId: number;
  detectionId?: number;
  time: string;
  fullTime?: string;
  camera: string;
  object: string;
  type: string;
  confidence: string;
  severity: Severity;
  status: string;
  statusLifecycle: string;
  picName?: string;
  maintenanceNote?: string;
  pvFlatness?: number | null;
  maxDeviation?: number | null;
  rawImagePath?: string | null;
  annotatedPath?: string | null;
};

export type EvidenceItem = {
  id: string;
  inspectionId: number;
  camera: string;
  title: string;
  severity: Severity;
  object: string;
  time: string;
  captureTime?: string;
  area: string;
  confidence: string;
  statusLifecycle?: string;
  picName?: string;
  maintenanceNote?: string;
  rawImagePath?: string | null;
  annotatedPath?: string | null;
  imageUrl?: string | null;
};

export type AnalyticsData = {
  weekly: { d: string; n: number }[];
  pareto: { label: string; n: number; p: string; barCls: string }[];
  totalIncidents: number;
};

/**
 * Mendapatkan URL gambar dari backend Flask image server.
 */
export function getImageUrl(path?: string | null): string | null {
  if (!path) return null;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  if (path.startsWith("/api/image")) {
    return `http://localhost:5000${path}`;
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
}): Promise<RecordRow[]> {
  const query = new URLSearchParams();
  if (params?.camera && params.camera !== "ALL") query.append("camera", params.camera);
  if (params?.severity && params.severity !== "ALL") query.append("severity", params.severity);
  if (params?.status && params.status !== "ALL") query.append("status", params.status);
  if (params?.limit) query.append("limit", params.limit.toString());

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
 * Mengirim aksi teknisi / PIC (Acknowledge atau Mark Resolved).
 */
export async function submitInspectionAction(
  inspectionId: number,
  action: "ACKNOWLEDGED" | "RESOLVED" | "UNACKNOWLEDGED",
  picName: string,
  maintenanceNote: string
): Promise<{ success: boolean; inspectionId: number; statusLifecycle: string }> {
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

  if (!res.ok) throw new Error("Gagal memperbarui status inspeksi");
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
