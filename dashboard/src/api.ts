import type { ScoreResponse } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function fetchScore(ticker: string, refresh = false): Promise<ScoreResponse> {
  const url = new URL(`${API_BASE_URL}/score/${encodeURIComponent(ticker)}`);
  if (refresh) url.searchParams.set("refresh", "true");

  const response = await fetch(url.toString());
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? `Request failed with status ${response.status}`);
  }
  return response.json();
}
