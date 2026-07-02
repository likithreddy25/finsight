import { useState } from "react";
import { fetchScore } from "./api";
import BreakdownChart from "./components/BreakdownChart";
import FilingsList from "./components/FilingsList";
import ScoreMeter from "./components/ScoreMeter";
import TickerSearch from "./components/TickerSearch";
import type { ScoreResponse } from "./types";

export default function App() {
  const [result, setResult] = useState<ScoreResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSearch(ticker: string) {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchScore(ticker);
      setResult(data);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>FinSight</h1>
          <p>Company financial health scoring from SEC EDGAR filings</p>
        </div>
      </header>

      <TickerSearch onSearch={handleSearch} loading={loading} />

      {error && <div className="error-banner">{error}</div>}

      {!result && !loading && !error && (
        <div className="empty-state">Search a ticker to see its FinSight health score.</div>
      )}

      {result && (
        <>
          <div className="top-row">
            <ScoreMeter score={result.score} ticker={result.ticker} />
            <div className="card">
              <h2>Rationale</h2>
              <p className="rationale">{result.rationale}</p>
              <div className="meta-row">
                <span>Model: {result.model}</span>
                <span>Generated: {new Date(result.generated_at).toLocaleString()}</span>
              </div>
            </div>
          </div>

          <BreakdownChart breakdown={result.breakdown} />
          <FilingsList filings={result.filings_used} />
        </>
      )}
    </div>
  );
}
