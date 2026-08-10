"""Tests for health score monitoring agent."""
from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from agent.monitor import HealthMonitor, PriorScore, ScoreChange, ScoreDatabase
from score.scorer import HealthScore, ScoreBreakdown


@pytest.fixture
def temp_db():
    """Fixture: temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name


@pytest.fixture
def sample_health_score():
    """Fixture: sample HealthScore object."""
    return HealthScore(
        ticker="AAPL",
        score=7.5,
        breakdown=ScoreBreakdown(
            revenue_trend=0.2,
            risk_language=0.3,
            executive_changes=0.1,
            filing_sentiment=0.4,
        ),
        rationale="Strong revenue growth with moderate risks.",
        filings_used=["10-K 2024-01-15"],
        generated_at=datetime.now(timezone.utc).isoformat(),
        model="test-model",
    )


def test_score_database_init(temp_db):
    """Verify ScoreDatabase creates schema on init."""
    db = ScoreDatabase(db_path=temp_db)

    with sqlite3.connect(temp_db) as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

    assert "score_history" in tables


def test_save_and_retrieve_score(temp_db, sample_health_score):
    """Verify saving and retrieving a score works correctly."""
    db = ScoreDatabase(db_path=temp_db)
    db.save_score(sample_health_score)

    prior = db.get_latest_score("AAPL")

    assert prior is not None
    assert prior.ticker == "AAPL"
    assert prior.score == 7.5
    assert prior.model == "test-model"


def test_get_nonexistent_score(temp_db):
    """Verify None is returned for ticker with no stored score."""
    db = ScoreDatabase(db_path=temp_db)
    result = db.get_latest_score("NONEXISTENT")

    assert result is None


def test_get_latest_score_most_recent(temp_db):
    """Verify get_latest_score returns the most recent entry."""
    db = ScoreDatabase(db_path=temp_db)

    score1 = HealthScore(
        ticker="AAPL",
        score=7.0,
        breakdown=ScoreBreakdown(0, 0, 0, 0),
        rationale="First",
        filings_used=[],
        generated_at="2024-01-01T10:00:00Z",
        model="model1",
    )
    score2 = HealthScore(
        ticker="AAPL",
        score=7.5,
        breakdown=ScoreBreakdown(0, 0, 0, 0),
        rationale="Second",
        filings_used=[],
        generated_at="2024-01-02T10:00:00Z",
        model="model2",
    )

    db.save_score(score1)
    db.save_score(score2)
    latest = db.get_latest_score("AAPL")

    assert latest.score == 7.5
    assert latest.timestamp == "2024-01-02T10:00:00Z"


def test_get_all_latest_scores(temp_db):
    """Verify retrieving latest scores for multiple tickers."""
    db = ScoreDatabase(db_path=temp_db)

    score_aapl = HealthScore(
        ticker="AAPL",
        score=7.0,
        breakdown=ScoreBreakdown(0, 0, 0, 0),
        rationale="AAPL",
        filings_used=[],
        generated_at="2024-01-01T10:00:00Z",
        model="model",
    )
    score_msft = HealthScore(
        ticker="MSFT",
        score=8.0,
        breakdown=ScoreBreakdown(0, 0, 0, 0),
        rationale="MSFT",
        filings_used=[],
        generated_at="2024-01-01T10:00:00Z",
        model="model",
    )

    db.save_score(score_aapl)
    db.save_score(score_msft)
    results = db.get_all_latest_scores(["AAPL", "MSFT", "TSLA"])

    assert len(results) == 2
    assert results["AAPL"].score == 7.0
    assert results["MSFT"].score == 8.0
    assert "TSLA" not in results


def test_score_change_first_time_monitoring(temp_db):
    """Verify ScoreChange for first-time scores shows as no-prior."""
    monitor = HealthMonitor(db_path=temp_db)
    monitor.scorer = MagicMock()

    score = HealthScore(
        ticker="AAPL",
        score=7.5,
        breakdown=ScoreBreakdown(0, 0, 0, 0),
        rationale="Test",
        filings_used=[],
        generated_at="2024-01-01T10:00:00Z",
        model="model",
    )
    monitor.scorer.score.return_value = score

    changes = monitor.monitor_watchlist(["AAPL"])

    assert len(changes) == 1
    assert changes[0].ticker == "AAPL"
    assert changes[0].prior_score is None
    assert changes[0].current_score == 7.5
    assert "First-time score" in changes[0].flagged_reason


def test_score_change_detected(temp_db):
    """Verify material score changes are detected."""
    monitor = HealthMonitor(db_path=temp_db)
    monitor.scorer = MagicMock()

    prior_score = HealthScore(
        ticker="AAPL",
        score=7.0,
        breakdown=ScoreBreakdown(0, 0, 0, 0),
        rationale="Prior",
        filings_used=[],
        generated_at="2024-01-01T10:00:00Z",
        model="model",
    )
    current_score = HealthScore(
        ticker="AAPL",
        score=5.3,
        breakdown=ScoreBreakdown(0, 0, 0, 0),
        rationale="Current",
        filings_used=[],
        generated_at="2024-01-02T10:00:00Z",
        model="model",
    )

    monitor.db.save_score(prior_score)
    monitor.scorer.score.return_value = current_score

    changes = monitor.monitor_watchlist(["AAPL"], score_threshold=1.5)

    assert len(changes) == 1
    assert changes[0].score_delta == pytest.approx(-1.7)
    assert "1.7" in changes[0].flagged_reason


def test_no_change_below_threshold(temp_db):
    """Verify small score changes below threshold are not flagged."""
    monitor = HealthMonitor(db_path=temp_db)
    monitor.scorer = MagicMock()

    prior_score = HealthScore(
        ticker="AAPL",
        score=7.0,
        breakdown=ScoreBreakdown(0, 0, 0, 0),
        rationale="Prior",
        filings_used=[],
        generated_at="2024-01-01T10:00:00Z",
        model="model",
    )
    current_score = HealthScore(
        ticker="AAPL",
        score=7.3,
        breakdown=ScoreBreakdown(0, 0, 0, 0),
        rationale="Current",
        filings_used=[],
        generated_at="2024-01-02T10:00:00Z",
        model="model",
    )

    monitor.db.save_score(prior_score)
    monitor.scorer.score.return_value = current_score

    changes = monitor.monitor_watchlist(["AAPL"], score_threshold=1.5)

    assert len(changes) == 0


def test_multiple_tickers_mixed_changes(temp_db):
    """Verify monitoring multiple tickers with mixed change statuses."""
    monitor = HealthMonitor(db_path=temp_db)
    monitor.scorer = MagicMock()

    prior_aapl = HealthScore(
        ticker="AAPL",
        score=7.0,
        breakdown=ScoreBreakdown(0, 0, 0, 0),
        rationale="",
        filings_used=[],
        generated_at="2024-01-01T10:00:00Z",
        model="model",
    )
    prior_msft = HealthScore(
        ticker="MSFT",
        score=8.0,
        breakdown=ScoreBreakdown(0, 0, 0, 0),
        rationale="",
        filings_used=[],
        generated_at="2024-01-01T10:00:00Z",
        model="model",
    )

    current_aapl = HealthScore(
        ticker="AAPL",
        score=5.2,
        breakdown=ScoreBreakdown(0, 0, 0, 0),
        rationale="",
        filings_used=[],
        generated_at="2024-01-02T10:00:00Z",
        model="model",
    )
    current_msft = HealthScore(
        ticker="MSFT",
        score=8.2,
        breakdown=ScoreBreakdown(0, 0, 0, 0),
        rationale="",
        filings_used=[],
        generated_at="2024-01-02T10:00:00Z",
        model="model",
    )

    monitor.db.save_score(prior_aapl)
    monitor.db.save_score(prior_msft)

    def score_side_effect(ticker):
        return current_aapl if ticker == "AAPL" else current_msft

    monitor.scorer.score.side_effect = score_side_effect

    changes = monitor.monitor_watchlist(["AAPL", "MSFT"], score_threshold=1.5)

    assert len(changes) == 1
    assert changes[0].ticker == "AAPL"


def test_score_change_dataclass():
    """Verify ScoreChange dataclass works correctly."""
    change = ScoreChange(
        ticker="AAPL",
        prior_score=7.0,
        current_score=5.5,
        score_delta=-1.5,
        flagged_reason="Test reason",
        prior_timestamp="2024-01-01T10:00:00Z",
        current_timestamp="2024-01-02T10:00:00Z",
    )

    assert change.ticker == "AAPL"
    assert change.score_delta == -1.5
    assert "Test reason" in change.flagged_reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
