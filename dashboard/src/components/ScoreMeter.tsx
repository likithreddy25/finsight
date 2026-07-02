import StatusBadge from "./StatusBadge";

interface ScoreMeterProps {
  score: number;
  ticker: string;
}

const MAX_SCORE = 10;

export default function ScoreMeter({ score, ticker }: ScoreMeterProps) {
  const pct = Math.max(0, Math.min(100, (score / MAX_SCORE) * 100));

  return (
    <div className="card">
      <h2>{ticker} Health Score</h2>
      <div style={{ fontSize: 48, fontWeight: 700, lineHeight: 1 }}>
        {score.toFixed(1)}
        <span style={{ fontSize: 18, fontWeight: 500, color: "var(--text-muted)" }}> / {MAX_SCORE}</span>
      </div>
      <div style={{ marginTop: 12, marginBottom: 10 }}>
        <StatusBadge score={score} />
      </div>
      <div
        role="meter"
        aria-valuenow={score}
        aria-valuemin={0}
        aria-valuemax={MAX_SCORE}
        style={{
          height: 8,
          borderRadius: 4,
          background: "var(--gridline)",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            borderRadius: 4,
            background: "var(--series-blue)",
            transition: "width 300ms ease",
          }}
        />
      </div>
    </div>
  );
}
