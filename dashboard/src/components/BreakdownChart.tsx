import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ScoreBreakdown } from "../types";

interface BreakdownChartProps {
  breakdown: ScoreBreakdown;
}

interface Row {
  label: string;
  value: number; // -1..1, positive = contributes toward a higher score
  raw: string;
}

function buildRows(b: ScoreBreakdown): Row[] {
  return [
    { label: "Revenue Trend", value: b.revenue_trend, raw: `${b.revenue_trend.toFixed(2)} raw (-1 shrinking .. +1 growing)` },
    { label: "Filing Sentiment", value: b.filing_sentiment, raw: `${b.filing_sentiment.toFixed(2)} raw (-1 negative .. +1 positive)` },
    { label: "Risk Language", value: -b.risk_language, raw: `${b.risk_language.toFixed(2)} raw (0 low .. 1 heavy risk language)` },
    { label: "Executive Changes", value: -b.executive_changes, raw: `${b.executive_changes.toFixed(2)} raw (0 stable .. 1 high churn)` },
  ];
}

export default function BreakdownChart({ breakdown }: BreakdownChartProps) {
  const data = buildRows(breakdown);

  return (
    <div className="card">
      <h2>Score Breakdown</h2>
      <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: -8, marginBottom: 16 }}>
        Contribution to score — right (blue) helps, left (red) hurts
      </p>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} layout="vertical" margin={{ left: 24, right: 24, top: 4, bottom: 4 }}>
          <XAxis
            type="number"
            domain={[-1, 1]}
            tick={{ fill: "var(--text-muted)", fontSize: 11 }}
            stroke="var(--baseline)"
          />
          <YAxis
            type="category"
            dataKey="label"
            width={130}
            tick={{ fill: "var(--text-secondary)", fontSize: 12 }}
            stroke="var(--baseline)"
            tickLine={false}
          />
          <ReferenceLine x={0} stroke="var(--baseline)" />
          <Tooltip
            cursor={{ fill: "var(--gridline)" }}
            contentStyle={{
              background: "var(--surface-1)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: "var(--text-primary)" }}
            formatter={(_value, _name, props) => [(props.payload as Row).raw, "signal"]}
          />
          <Bar dataKey="value" radius={4} maxBarSize={22}>
            {data.map((row) => (
              <Cell key={row.label} fill={row.value >= 0 ? "var(--series-blue)" : "var(--series-red)"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
