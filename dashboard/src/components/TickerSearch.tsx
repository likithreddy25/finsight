import { FormEvent, useState } from "react";

interface TickerSearchProps {
  onSearch: (ticker: string) => void;
  loading: boolean;
}

export default function TickerSearch({ onSearch, loading }: TickerSearchProps) {
  const [value, setValue] = useState("");

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const ticker = value.trim().toUpperCase();
    if (ticker) onSearch(ticker);
  }

  return (
    <form className="search-row" onSubmit={handleSubmit}>
      <input
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="Enter a ticker, e.g. AAPL"
        aria-label="Ticker symbol"
        autoFocus
      />
      <button type="submit" disabled={loading || value.trim() === ""}>
        {loading ? "Scoring…" : "Get Score"}
      </button>
    </form>
  );
}
