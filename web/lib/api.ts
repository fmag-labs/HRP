// Typed HTTP client for the HRP advisory API.
// Base URL and optional bearer token come from public env vars.
// The API contract is frozen — see web/README.md.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:8090";

const API_TOKEN = process.env.NEXT_PUBLIC_API_TOKEN;

// ---- Domain types (frozen contract) ----

export type Action = "BUY" | "HOLD" | "SELL" | string;

export interface Recommendation {
  recommendation_id: string;
  symbol: string;
  action: Action;
  confidence: string;
  signal_strength: number;
  entry_price: number | null;
  close_price: number | null;
  realized_return: number | null;
  status: string;
  created_at: string | null;
  closed_at: string | null;
}

export interface RecommendationProvenance {
  hypothesis_id: string | null;
  model_name: string | null;
  validation_status: string | null;
}

export interface RecommendationDetail {
  recommendation_id: string;
  symbol: string;
  action: Action;
  confidence: string;
  signal_strength: number;
  entry_price: number | null;
  target_price: number | null;
  stop_price: number | null;
  position_pct: number | null;
  thesis: string | null;
  risks: string | null;
  time_horizon_days: number | null;
  status: string;
  created_at: string | null;
  closed_at: string | null;
  provenance: RecommendationProvenance;
}

export interface ApprovalResult {
  recommendation_id: string;
  action: string;
  actor: string;
  order_id?: string | null;
  message: string;
}

export interface RejectionResult {
  recommendation_id: string;
  action: string;
  actor: string;
  reason?: string | null;
  message: string;
}

export interface Position {
  symbol: string;
  quantity: number;
  avg_cost: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
}

export interface Status {
  ok: boolean;
  message: string;
  data: { last_date: string | null; days_stale: number | null; is_fresh: boolean };
  symbol_count: number;
  recommendation_count: number;
  position_count: number;
}

export interface Portfolio {
  total_value: number;
  position_count: number;
  positions: Position[];
}

export interface TrackRecordPeriod {
  period_start: string;
  period_end: string;
  total_recommendations: number;
  profitable: number;
  unprofitable: number;
  closed_recommendations: number;
  win_rate: number;
  avg_return: number;
  avg_win: number;
  avg_loss: number;
  best_pick: string | null;
  worst_pick: string | null;
  benchmark_return: number;
  excess_return: number;
}

export interface AssistantResponse {
  answer: string;
  remaining_today: number;
  grounded_on: string[];
  model?: string;
}

export interface ModelInfo {
  key: string;
  label: string;
  model: string;
  available: boolean;
}

export type Horizon = "short" | "medium" | "long";

export interface Settings {
  profile_id: string;
  name: string;
  risk_tolerance: number;
  portfolio_value: number;
  max_positions: number;
  max_position_pct: number;
  excluded_symbols: string[];
  excluded_sectors: string[];
  preferred_horizon: Horizon;
}

// PUT /api/settings accepts any subset of the editable fields.
export interface SettingsUpdate {
  name?: string;
  risk_tolerance?: number;
  portfolio_value?: number;
  max_positions?: number;
  max_position_pct?: number;
  excluded_symbols?: string[];
  excluded_sectors?: string[];
  preferred_horizon?: Horizon;
}

// ---- Research screens ----

export interface ScreenInfo {
  key: string;
  title: string;
  subtitle: string;
  value_label: string;
}

export interface ScreenRow {
  rank: number;
  symbol: string;
  name: string | null;
  sector: string | null;
  value: number | null;
}

export interface ScreenResult {
  screen: string;
  title: string;
  subtitle: string;
  value_label: string;
  as_of: string | null;
  rows: ScreenRow[];
}

// ---- Error type ----

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

// ---- Core request helper ----

function buildHeaders(hasBody: boolean): HeadersInit {
  const headers: Record<string, string> = {};
  if (hasBody) headers["Content-Type"] = "application/json";
  if (API_TOKEN) headers["Authorization"] = `Bearer ${API_TOKEN}`;
  return headers;
}

async function request<T>(
  path: string,
  options: { method?: string; body?: unknown } = {}
): Promise<T> {
  const { method = "GET", body } = options;
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method,
      headers: buildHeaders(body !== undefined),
      body: body !== undefined ? JSON.stringify(body) : undefined,
      cache: "no-store",
    });
  } catch {
    throw new ApiError(
      `Could not reach the research terminal at ${API_BASE}. Is the API running?`,
      0
    );
  }

  if (!res.ok) {
    let detail = "";
    try {
      const data = await res.json();
      detail = (data && (data.detail || data.message || data.error)) || "";
    } catch {
      // ignore non-JSON error bodies
    }
    throw new ApiError(
      detail || `Request failed (${res.status} ${res.statusText})`,
      res.status
    );
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// ---- Query string helper ----

function qs(params: Record<string, string | number | undefined | null>): string {
  const parts = Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`);
  return parts.length ? `?${parts.join("&")}` : "";
}

// ---- Endpoint functions ----

export const api = {
  getRecommendations(params: {
    status?: string;
    symbol?: string;
    limit?: number;
  } = {}): Promise<Recommendation[]> {
    return request<Recommendation[]>(`/api/recommendations${qs(params)}`);
  },

  getRecommendationHistory(limit?: number): Promise<Recommendation[]> {
    return request<Recommendation[]>(`/api/recommendations/history${qs({ limit })}`);
  },

  getRecommendation(id: string): Promise<RecommendationDetail> {
    return request<RecommendationDetail>(`/api/recommendations/${encodeURIComponent(id)}`);
  },

  approveRecommendation(
    id: string,
    body: { actor?: string; dry_run?: boolean } = {}
  ): Promise<ApprovalResult> {
    return request<ApprovalResult>(
      `/api/recommendations/${encodeURIComponent(id)}/approve`,
      { method: "POST", body }
    );
  },

  rejectRecommendation(
    id: string,
    body: { actor?: string; reason?: string } = {}
  ): Promise<RejectionResult> {
    return request<RejectionResult>(
      `/api/recommendations/${encodeURIComponent(id)}/reject`,
      { method: "POST", body }
    );
  },

  approveAll(
    body: { actor?: string; dry_run?: boolean } = {}
  ): Promise<ApprovalResult[]> {
    return request<ApprovalResult[]>(`/api/recommendations/approve-all`, {
      method: "POST",
      body,
    });
  },

  getPortfolio(): Promise<Portfolio> {
    return request<Portfolio>(`/api/portfolio`);
  },

  getStatus(): Promise<Status> {
    return request<Status>(`/api/status`);
  },

  getTrackRecord(params: { start?: string; end?: string } = {}): Promise<
    TrackRecordPeriod[]
  > {
    return request<TrackRecordPeriod[]>(`/api/track-record${qs(params)}`);
  },

  getAssistantModels(): Promise<ModelInfo[]> {
    return request<ModelInfo[]>(`/api/assistant/models`);
  },

  askAssistant(question: string, model?: string): Promise<AssistantResponse> {
    return request<AssistantResponse>(`/api/assistant/query`, {
      method: "POST",
      body: { question, model },
    });
  },

  getSettings(): Promise<Settings> {
    return request<Settings>(`/api/settings`);
  },

  updateSettings(body: SettingsUpdate): Promise<Settings> {
    return request<Settings>(`/api/settings`, { method: "PUT", body });
  },

  getScreens(): Promise<ScreenInfo[]> {
    return request<ScreenInfo[]>(`/api/screens`);
  },

  getScreen(key: string, limit?: number): Promise<ScreenResult> {
    return request<ScreenResult>(
      `/api/screens/${encodeURIComponent(key)}${qs({ limit })}`
    );
  },
};
