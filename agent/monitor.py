"""Scheduled monitoring agent: track health scores for a watchlist of companies.

Runs scoring for a watchlist of tickers, compares against prior scores stored in a
state database (SQLite or JSON), and flags material changes for investigation.

Can be run standalone: python -m agent.monitor --watchlist AAPL MSFT TSLA
or scheduled via cron: 0 9 * * * cd /path && python -m agent.monitor --watchlist ...
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from score.scorer import FinancialHealthScorer, HealthScore

logger = logging.getLogger(__name__)

MONITORING_DB = "data/monitoring.db"
SCORE_THRESHOLD = 1.5


@dataclass
class PriorScore:
    """Prior snapshot of a company's health score."""

    ticker: str
    score: float
    timestamp: str
    rationale: str
    model: str


@dataclass
class ScoreChange:
    """A detected material change in a company's health score."""

    ticker: str
    prior_score: float | None
    current_score: float
    score_delta: float
    flagged_reason: str
    prior_timestamp: str | None
    current_timestamp: str


class ScoreDatabase:
    """Persistent storage for prior company health scores."""

    def __init__(self, db_path: str = MONITORING_DB):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create schema if it doesn't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS score_history (
                    ticker TEXT NOT NULL,
                    score REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    rationale TEXT,
                    model TEXT,
                    PRIMARY KEY (ticker, timestamp)
                )
                """
            )
            conn.commit()

    def save_score(self, health_score: HealthScore) -> None:
        """Store a health score in the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO score_history (ticker, score, timestamp, rationale, model) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    health_score.ticker,
                    health_score.score,
                    health_score.generated_at,
                    health_score.rationale[:500],
                    health_score.model,
                ),
            )
            conn.commit()

    def get_latest_score(self, ticker: str) -> PriorScore | None:
        """Retrieve the most recent stored score for a ticker."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT ticker, score, timestamp, rationale, model FROM score_history "
                "WHERE ticker = ? ORDER BY timestamp DESC LIMIT 1",
                (ticker,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        return PriorScore(
            ticker=row[0],
            score=row[1],
            timestamp=row[2],
            rationale=row[3] or "",
            model=row[4] or "",
        )

    def get_all_latest_scores(self, tickers: list[str]) -> dict[str, PriorScore]:
        """Retrieve latest scores for multiple tickers."""
        return {
            ticker: score
            for ticker in tickers
            if (score := self.get_latest_score(ticker)) is not None
        }


class HealthMonitor:
    """Monitors company health scores and flags material changes."""

    def __init__(self, db_path: str = MONITORING_DB):
        self.scorer = FinancialHealthScorer()
        self.db = ScoreDatabase(db_path)

    def monitor_watchlist(
        self, tickers: list[str], score_threshold: float = SCORE_THRESHOLD
    ) -> list[ScoreChange]:
        """Score a watchlist of tickers and flag material changes.

        Args:
            tickers: List of company tickers to monitor
            score_threshold: Score change magnitude to flag (default 1.5 points)

        Returns:
            List of ScoreChange objects for tickers with material changes"""

        changes: list[ScoreChange] = []
        prior_scores = self.db.get_all_latest_scores(tickers)

        for ticker in tickers:
            logger.info("Monitoring %s...", ticker)
            try:
                current_score = self.scorer.score(ticker)
            except Exception:
                logger.exception("Failed to score %s", ticker)
                continue

            self.db.save_score(current_score)

            prior = prior_scores.get(ticker)
            if prior is None:
                changes.append(
                    ScoreChange(
                        ticker=ticker,
                        prior_score=None,
                        current_score=current_score.score,
                        score_delta=0.0,
                        flagged_reason="First-time score (no prior baseline)",
                        prior_timestamp=None,
                        current_timestamp=current_score.generated_at,
                    )
                )
            else:
                delta = abs(current_score.score - prior.score)
                if delta >= score_threshold:
                    reason = (
                        f"Score changed {current_score.score - prior.score:+.1f} points "
                        f"(was {prior.score}, now {current_score.score})"
                    )
                    changes.append(
                        ScoreChange(
                            ticker=ticker,
                            prior_score=prior.score,
                            current_score=current_score.score,
                            score_delta=current_score.score - prior.score,
                            flagged_reason=reason,
                            prior_timestamp=prior.timestamp,
                            current_timestamp=current_score.generated_at,
                        )
                    )

        return changes


def format_changes_report(changes: list[ScoreChange]) -> str:
    """Format score changes as a human-readable report."""
    if not changes:
        return "No material changes detected."

    lines = [f"Detected {len(changes)} material change(s):\n"]
    for change in changes:
        lines.append(f"  {change.ticker}:")
        if change.prior_score is not None:
            lines.append(f"    Prior score: {change.prior_score} ({change.prior_timestamp})")
        lines.append(f"    Current score: {change.current_score} ({change.current_timestamp})")
        if change.prior_score is not None:
            lines.append(f"    Delta: {change.score_delta:+.1f} points")
        lines.append(f"    Reason: {change.flagged_reason}\n")

    return "".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Monitor company health scores and flag material changes."
    )
    parser.add_argument(
        "--watchlist",
        nargs="+",
        required=True,
        help="Tickers to monitor, e.g. AAPL MSFT TSLA",
    )
    parser.add_argument(
        "--db",
        default=MONITORING_DB,
        help="Path to SQLite score database (default: data/monitoring.db)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=SCORE_THRESHOLD,
        help=f"Score change threshold to flag (default: {SCORE_THRESHOLD})",
    )
    parser.add_argument(
        "--output-json",
        help="Write results as JSON to this file (for automation)",
    )
    args = parser.parse_args()

    monitor = HealthMonitor(db_path=args.db)
    changes = monitor.monitor_watchlist(args.watchlist, score_threshold=args.threshold)

    print("\n" + "=" * 80)
    print("HEALTH MONITORING REPORT")
    print(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)
    print()
    print(format_changes_report(changes))

    if args.output_json:
        output_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_monitored": len(args.watchlist),
            "n_changes": len(changes),
            "changes": [asdict(c) for c in changes],
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nJSON report saved to {args.output_json}")
