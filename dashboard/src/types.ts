export interface ScoreBreakdown {
  revenue_trend: number; // -1..1, positive = growing
  risk_language: number; // 0..1, higher = more risk language
  executive_changes: number; // 0..1, higher = more leadership churn
  filing_sentiment: number; // -1..1, positive = more positive tone
}

export interface ScoreResponse {
  ticker: string;
  score: number; // 1..10
  breakdown: ScoreBreakdown;
  rationale: string;
  filings_used: string[];
  generated_at: string;
  model: string;
}

export interface ApiError {
  detail: string;
}
