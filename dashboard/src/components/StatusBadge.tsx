interface StatusBadgeProps {
  score: number;
}

function statusFor(score: number): { label: string; color: string } {
  if (score >= 7) return { label: "Healthy", color: "var(--status-good)" };
  if (score >= 4) return { label: "Watch", color: "var(--status-warning)" };
  return { label: "At Risk", color: "var(--status-critical)" };
}

export default function StatusBadge({ score }: StatusBadgeProps) {
  const { label, color } = statusFor(score);
  return (
    <span className="status-pill" style={{ background: `color-mix(in srgb, ${color} 16%, transparent)`, color }}>
      <span className="dot" style={{ background: color }} />
      {label}
    </span>
  );
}
