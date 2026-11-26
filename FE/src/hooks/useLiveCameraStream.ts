import { useCallback, useEffect, useRef, useState } from "react";

export type CameraId = "CAM-01" | "CAM-02" | "CAM-03";

export interface CameraStreamMeta {
  cam: CameraId;
  status: string;
  fps: number;
  object?: string;
  conf?: number | null;
  severity?: string;
  slatNo?: number | null;
  ts?: number;
}

export interface CameraStreamState {
  frameUrl: string | null;
  meta: CameraStreamMeta | null;
  lastUpdatedAt: number;
}

const EMPTY_STREAM: CameraStreamState = {
  frameUrl: null,
  meta: null,
  lastUpdatedAt: 0,
};

/**
 * Parse a binary WebSocket packet packed as:
 * `[4-byte Big-Endian uint32 metaLen][JSON header][JPEG bytes]`.
 */
export function parseWsBinaryPacket(
  buffer: ArrayBuffer
): { meta: CameraStreamMeta; blob: Blob } | null {
  if (!buffer || buffer.byteLength < 4) return null;
  try {
    const view = new DataView(buffer);
    const metaLen = view.getUint32(0, false);
    if (metaLen <= 0 || buffer.byteLength < 4 + metaLen) return null;

    const metaBytes = new Uint8Array(buffer, 4, metaLen);
    const meta = JSON.parse(new TextDecoder("utf-8").decode(metaBytes)) as CameraStreamMeta;
    const jpegBytes = new Uint8Array(buffer, 4 + metaLen);
    const blob = new Blob([jpegBytes], { type: "image/jpeg" });
    return { meta, blob };
  } catch {
    return null;
  }
}

export function useLiveCameraStream(enabled: boolean, port = 8765) {
  const [connected, setConnected] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [dbEnabled, setDbEnabled] = useState<boolean | null>(null);
  const [streams, setStreams] = useState<Record<CameraId, CameraStreamState>>({
    "CAM-01": EMPTY_STREAM,
    "CAM-02": EMPTY_STREAM,
    "CAM-03": EMPTY_STREAM,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const urlsRef = useRef<Record<CameraId, string | null>>({
    "CAM-01": null,
    "CAM-02": null,
    "CAM-03": null,
  });
  const pendingRef = useRef<Partial<Record<CameraId, CameraStreamState>>>({});
  const rafRef = useRef<number | null>(null);
  const showOverlayRef = useRef(showOverlay);
  showOverlayRef.current = showOverlay;

  const revokeAllUrls = useCallback(() => {
    if (rafRef.current !== null && typeof cancelAnimationFrame === "function") {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    (["CAM-01", "CAM-02", "CAM-03"] as CameraId[]).forEach((camId) => {
      const pend = pendingRef.current[camId];
      if (pend?.frameUrl) {
        URL.revokeObjectURL(pend.frameUrl);
      }
      delete pendingRef.current[camId];
      const prev = urlsRef.current[camId];
      if (prev) {
        URL.revokeObjectURL(prev);
        urlsRef.current[camId] = null;
      }
    });
  }, []);

  useEffect(() => {
    if (!enabled) {
      wsRef.current?.close();
      wsRef.current = null;
      setConnected(false);
      return;
    }

    let disposed = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (disposed || document.hidden) return;
      if (wsRef.current && wsRef.current.readyState <= WebSocket.OPEN) return;

      const host = typeof window !== "undefined" ? window.location.hostname : "localhost";
      const ws = new WebSocket(`ws://${host}:${port}`);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        if (disposed) {
          ws.close();
          return;
        }
        setConnected(true);
        ws.send(JSON.stringify({ show_overlay: showOverlayRef.current }));
      };

      ws.onmessage = (event) => {
        if (disposed) return;
        if (typeof event.data === "string") {
          try {
            const msg = JSON.parse(event.data) as {
              type?: string;
              show_overlay?: boolean;
              db_enabled?: boolean;
            };
            if (msg?.type === "state") {
              if (typeof msg.show_overlay === "boolean") {
                setShowOverlay(msg.show_overlay);
              }
              if (typeof msg.db_enabled === "boolean") {
                setDbEnabled(msg.db_enabled);
              }
            }
          } catch {
            // ignore malformed control JSON
          }
          return;
        }
        if (!(event.data instanceof ArrayBuffer)) return;
        const parsed = parseWsBinaryPacket(event.data);
        if (!parsed) return;

        const { meta, blob } = parsed;
        const camId = meta.cam;
        if (camId !== "CAM-01" && camId !== "CAM-02" && camId !== "CAM-03") return;

        const nextUrl = URL.createObjectURL(blob);
        const superseded = pendingRef.current[camId];
        if (superseded?.frameUrl) {
          URL.revokeObjectURL(superseded.frameUrl);
        }
        pendingRef.current[camId] = {
          frameUrl: nextUrl,
          meta,
          lastUpdatedAt: Date.now(),
        };

        if (rafRef.current === null) {
          const flush = () => {
            rafRef.current = null;
            if (disposed) return;
            const batch = pendingRef.current;
            const keys = Object.keys(batch) as CameraId[];
            if (keys.length === 0) return;
            pendingRef.current = {};

            for (const k of keys) {
              const entry = batch[k];
              if (!entry) continue;
              const prevCommitted = urlsRef.current[k];
              urlsRef.current[k] = entry.frameUrl;
              if (prevCommitted && prevCommitted !== entry.frameUrl) {
                URL.revokeObjectURL(prevCommitted);
              }
            }

            setStreams((prev) => {
              const next = { ...prev };
              for (const k of keys) {
                const entry = batch[k];
                if (entry) next[k] = entry;
              }
              return next;
            });
          };

          if (typeof requestAnimationFrame === "function") {
            rafRef.current = requestAnimationFrame(flush);
          } else {
            flush();
          }
        }
      };

      ws.onclose = () => {
        if (wsRef.current === ws) {
          wsRef.current = null;
        }
        setConnected(false);
        setDbEnabled(null);
        if (!disposed && !document.hidden) {
          reconnectTimer = setTimeout(connect, 2000);
        }
      };

      ws.onerror = () => {
        ws.close();
      };
    };

    const handleVisibilityChange = () => {
      if (document.hidden) {
        if (reconnectTimer) {
          clearTimeout(reconnectTimer);
          reconnectTimer = null;
        }
        wsRef.current?.close();
        wsRef.current = null;
        setConnected(false);
      } else {
        connect();
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    connect();

    return () => {
      disposed = true;
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      if (reconnectTimer) clearTimeout(reconnectTimer);
      wsRef.current?.close();
      wsRef.current = null;
      setConnected(false);
      setDbEnabled(null);
      revokeAllUrls();
      setStreams({
        "CAM-01": EMPTY_STREAM,
        "CAM-02": EMPTY_STREAM,
        "CAM-03": EMPTY_STREAM,
      });
    };
  }, [enabled, port, revokeAllUrls]);

  const toggleOverlay = useCallback(() => {
    setShowOverlay((prev) => {
      const next = !prev;
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ show_overlay: next }));
      }
      return next;
    });
  }, []);

  const toggleDb = useCallback(() => {
    setDbEnabled((prev) => {
      const next = !(prev ?? false);
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ db_enabled: next }));
      }
      return next;
    });
  }, []);

  return {
    connected,
    showOverlay,
    toggleOverlay,
    dbEnabled,
    toggleDb,
    streams,
  };
}
