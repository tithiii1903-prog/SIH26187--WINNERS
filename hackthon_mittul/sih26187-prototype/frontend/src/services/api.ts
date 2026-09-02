const rawUrl = import.meta.env.VITE_API_BASE_URL || "https://sih26187-winners-production.up.railway.app";
let sanitizedUrl = rawUrl.replace(/\/+$/, "");
if (typeof window !== "undefined" && window.location.protocol === "https:" && sanitizedUrl.startsWith("http://")) {
  sanitizedUrl = sanitizedUrl.replace(/^http:\/\//, "https://");
}
export const API_BASE_URL = sanitizedUrl;

// ============================================================
// NEW LIVE FEED TYPES
// ============================================================

export interface Feed {
  id: string;
  name: string;
  source_type?: 'file' | 'camera';
  device_index?: number;
  filename: string;
  filepath: string;
  width: number;
  height: number;
  fps: number;
  frame_count: number | null;
  duration: number | null;
  status: string;
}

export interface WatchlistMatch {
  track_id: number;
  name: string;
  status: string;
  similarity: number;
  wl_id: string;
}

export interface LiveAnalytics {
  current_persons: number;
  active_tracks: number;
  peak_persons: number;
  current_vehicles: number;
  peak_vehicles: number;
  cars: number;
  motorcycles: number;
  buses: number;
  trucks: number;
  active_intrusions: number[];
  total_intrusion_entries: number;
  total_intrusion_exits: number;
  processing_fps: number;
  stream_fps?: number;
  frames_processed: number;
  max_active_tracks: number;
  source_fps: number;
  timestamp: number;
  status: string;
  modules?: ModuleConfig;
  active_watchlist_matches?: WatchlistMatch[];
}

export interface ModuleConfig {
  human_detection: boolean;
  human_tracking: boolean;
  vehicle_detection: boolean;
  virtual_fence: boolean;
}

export interface LiveEvent {
  timestamp: number;
  type: string;
  track_id?: number;
  zone_id?: string;
  zone_name?: string;
  description: string;
  wl_id?: string;
  wl_name?: string;
  wl_status?: string;
  similarity?: number;
}

// ============================================================
// WATCHLIST TYPES
// ============================================================

export interface WatchlistRecord {
  id: string;
  name: string;
  status: string;
  created: number;
  enabled: boolean;
}

// ============================================================
// NEW LIVE FEED API FUNCTIONS
// ============================================================

export const fetchFeeds = async (): Promise<Feed[]> => {
  const res = await fetch(`${API_BASE_URL}/api/feeds`);
  if (!res.ok) throw new Error("Failed to fetch feeds");
  const data = await res.json();
  return data.feeds || [];
};

export const uploadFeed = async (file: File, name: string): Promise<Feed> => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("name", name);

  const res = await fetch(`${API_BASE_URL}/api/feeds`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to upload feed");
  }
  return res.json();
};

export const createCameraFeed = async (name: string, deviceIndex: number = 0): Promise<Feed> => {
  const res = await fetch(`${API_BASE_URL}/api/feeds/camera`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, device_index: deviceIndex }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to register camera");
  }
  return res.json();
};

export const deleteFeed = async (feedId: string): Promise<void> => {
  const res = await fetch(`${API_BASE_URL}/api/feeds/${feedId}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to delete feed");
  }
};

export const startFeed = async (feedId: string): Promise<Feed> => {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/api/feeds/${feedId}/start`, {
      method: "POST",
    });
  } catch (err: any) {
    throw new Error(err.message || "Failed to reach server (CORS or offline)");
  }

  if (!res.ok) {
    let errMsg = "Failed to start feed";
    try {
      const err = await res.json();
      errMsg = err.detail || errMsg;
    } catch {
      const text = await res.text().catch(() => "");
      if (text) errMsg = `Server Error: ${text}`;
    }
    throw new Error(errMsg);
  }
  const data = await res.json();
  return data.feed;
};

export const stopFeed = async (feedId: string): Promise<Feed> => {
  const res = await fetch(`${API_BASE_URL}/api/feeds/${feedId}/stop`, {
    method: "POST",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to stop feed");
  }
  const data = await res.json();
  return data.feed;
};

export const fetchFeedAnalytics = async (feedId: string): Promise<LiveAnalytics> => {
  const res = await fetch(`${API_BASE_URL}/api/feeds/${feedId}/analytics`);
  if (!res.ok) throw new Error("Failed to fetch feed analytics");
  return res.json();
};

export const fetchFeedEvents = async (feedId: string): Promise<LiveEvent[]> => {
  const res = await fetch(`${API_BASE_URL}/api/feeds/${feedId}/events`);
  if (!res.ok) throw new Error("Failed to fetch feed events");
  const data = await res.json();
  return data.events || [];
};

export const updateModules = async (feedId: string, modules: Partial<ModuleConfig>): Promise<ModuleConfig> => {
  const res = await fetch(`${API_BASE_URL}/api/feeds/${feedId}/modules`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(modules),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to update modules");
  }
  const data = await res.json();
  return data.modules;
};

export const getMjpegStreamUrl = (feedId: string, sessionId?: string | number): string => {
  if (sessionId !== undefined && sessionId !== null && sessionId !== '') {
    return `${API_BASE_URL}/api/stream/${feedId}?session=${sessionId}`;
  }
  return `${API_BASE_URL}/api/stream/${feedId}`;
};

// ============================================================
// ZONE API
// ============================================================

export interface Zone {
  id: string;
  name: string;
  enabled: boolean;
  polygon: number[][];
}

export const fetchZones = async (): Promise<{ zones: Zone[] }> => {
  const res = await fetch(`${API_BASE_URL}/api/zones`);
  if (!res.ok) throw new Error("Failed to fetch zones");
  return res.json();
};

export const saveZones = async (name: string, enabled: boolean, polygon: number[][]): Promise<any> => {
  const res = await fetch(`${API_BASE_URL}/api/zones`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, enabled, polygon }),
  });
  if (!res.ok) throw new Error("Failed to save zones");
  return res.json();
};

// ============================================================
// EVENT ACKNOWLEDGEMENT
// ============================================================

export const acknowledgeEvent = async (eventId: string): Promise<any> => {
  const res = await fetch(`${API_BASE_URL}/api/events/${eventId}/acknowledge`, {
    method: "POST",
  });
  if (!res.ok) throw new Error("Failed to acknowledge event");
  return res.json();
};

// ============================================================
// EVENT ACKNOWLEDGEMENT QUERIES
// ============================================================

export const fetchAcknowledgedEvents = async (): Promise<{ acknowledged_events: string[] }> => {
  const res = await fetch(`${API_BASE_URL}/api/events/acknowledgements`);
  if (!res.ok) throw new Error("Failed to fetch acknowledged events");
  return res.json();
};

// ============================================================
// WATCHLIST API
// ============================================================

export const fetchWatchlist = async (): Promise<WatchlistRecord[]> => {
  const res = await fetch(`${API_BASE_URL}/api/watchlist`);
  if (!res.ok) throw new Error("Failed to fetch watchlist");
  const data = await res.json();
  return data.records || [];
};

export const enrollWatchlist = async (
  name: string,
  status: string,
  photo: File,
): Promise<WatchlistRecord> => {
  const formData = new FormData();
  formData.append("name", name);
  formData.append("status", status);
  formData.append("photo", photo);

  const res = await fetch(`${API_BASE_URL}/api/watchlist`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to enroll person");
  }
  return res.json();
};

export const enableWatchlistRecord = async (wlId: string): Promise<void> => {
  const res = await fetch(`${API_BASE_URL}/api/watchlist/${wlId}/enable`, {
    method: "POST",
  });
  if (!res.ok) throw new Error("Failed to enable record");
};

export const disableWatchlistRecord = async (wlId: string): Promise<void> => {
  const res = await fetch(`${API_BASE_URL}/api/watchlist/${wlId}/disable`, {
    method: "POST",
  });
  if (!res.ok) throw new Error("Failed to disable record");
};

export const deleteWatchlistRecord = async (wlId: string): Promise<void> => {
  const res = await fetch(`${API_BASE_URL}/api/watchlist/${wlId}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to delete record");
  }
};

// ============================================================
// HD FACE CAMERA API (Phase 13)
// ============================================================

export interface FaceDetection {
  face_id: number;
  bbox: number[];
  name: string | null;
  status: string | null;
  similarity: number;
  matched: boolean;
  confidence: number;
}

export interface FaceCameraResults {
  is_running: boolean;
  faces: FaceDetection[];
  fps: number;
  latency_ms: number;
  timestamp: number;
}

export interface FaceCameraStatus {
  is_running: boolean;
  device_index: number;
  source_fps: number;
  recognition_fps: number;
  average_recognition_latency_ms: number;
  number_of_faces: number;
  registered_faces_count: number;
  match_threshold: number;
  model_load_time_ms: number;
  resolution: [number, number];
}

export const startFaceCamera = async (deviceIndex: number = 0): Promise<{ status: string; device_index: number }> => {
  const res = await fetch(`${API_BASE_URL}/api/face-camera/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ device_index: deviceIndex }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to start HD Face Camera");
  }
  return res.json();
};

export const stopFaceCamera = async (): Promise<{ status: string }> => {
  const res = await fetch(`${API_BASE_URL}/api/face-camera/stop`, {
    method: "POST",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to stop HD Face Camera");
  }
  return res.json();
};

export const fetchFaceCameraStatus = async (): Promise<FaceCameraStatus> => {
  const res = await fetch(`${API_BASE_URL}/api/face-camera/status`);
  if (!res.ok) throw new Error("Failed to fetch HD Face Camera status");
  return res.json();
};

export const fetchFaceCameraResults = async (): Promise<FaceCameraResults> => {
  const res = await fetch(`${API_BASE_URL}/api/face-camera/results`);
  if (!res.ok) throw new Error("Failed to fetch HD Face Camera results");
  return res.json();
};

export const getFaceCameraStreamUrl = (): string => {
  return `${API_BASE_URL}/api/face-camera/stream`;
};


