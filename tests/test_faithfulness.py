"""Tests for faithfulness evaluation harness."""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from eval.faithfulness import FaithfulnessRow, summarize_faithfulness


def test_faithfulness_row_creation():
    """Verify FaithfulnessRow dataclass works correctly."""
    row = FaithfulnessRow(
        ticker="AAPL",
        query="What is the revenue?",
        answer="Revenue was $100B",
        faithfulness_score=0.75,
        grounded_answer="Revenue was $100B [partially grounded]",
        num_claims=2,
        num_supported=1,
    )

    assert row.ticker == "AAPL"
    assert row.faithfulness_score == 0.75
    assert row.num_claims == 2
    assert row.num_supported == 1
    assert row.error == ""


def test_faithfulness_row_with_error():
    """Verify FaithfulnessRow can record errors."""
    row = FaithfulnessRow(
        ticker="INVALID",
        query="Test query",
        answer="",
        faithfulness_score=0.0,
        grounded_answer="",
        num_claims=0,
        num_supported=0,
        error="No filings found for ticker",
    )

    assert row.error != ""
    assert row.faithfulness_score == 0.0


def test_summarize_faithfulness_empty_dataframe():
    """Verify handling of empty results."""
    df = pd.DataFrame(columns=["ticker", "query", "faithfulness_score", "num_claims", "num_supported", "error"])
    summary = summarize_faithfulness(df)

    assert summary["n_queries"] == 0
    assert "error" in summary


def test_summarize_faithfulness_with_valid_results():
    """Verify faithfulness statistics calculation."""
    rows = [
        FaithfulnessRow("AAPL", "Q1", "A1", 1.0, "A1", 2, 2),
        FaithfulnessRow("AAPL", "Q2", "A2", 0.5, "A2", 2, 1),
        FaithfulnessRow("MSFT", "Q1", "A3", 0.75, "A3", 4, 3),
    ]
    df = pd.DataFrame([vars(r) for r in rows])

    summary = summarize_faithfulness(df)

    assert summary["n_queries"] == 3
    assert summary["n_errors"] == 0
    assert summary["mean_faithfulness"] == pytest.approx(0.75, rel=0.01)
    assert summary["median_faithfulness"] == pytest.approx(0.75, rel=0.01)
    assert summary["max_faithfulness"] == 1.0
    assert summary["min_faithfulness"] == 0.5


def test_summarize_faithfulness_with_errors():
    """Verify error handling in summary statistics."""
    rows = [
        FaithfulnessRow("AAPL", "Q1", "A1", 0.8, "A1", 2, 1),
        FaithfulnessRow("MSFT", "Q1", "", 0.0, "", 0, 0, error="Agent unavailable"),
    ]
    df = pd.DataFrame([vars(r) for r in rows])

    summary = summarize_faithfulness(df)

    assert summary["n_queries"] == 2
    assert summary["n_errors"] == 1
    assert summary["mean_faithfulness"] == pytest.approx(0.8)


def test_summarize_by_ticker():
    """Verify per-ticker breakdown in summary."""
    rows = [
        FaithfulnessRow("AAPL", "Q1", "A1", 0.9, "A1", 2, 1),
        FaithfulnessRow("AAPL", "Q2", "A2", 0.8, "A2", 2, 1),
        FaithfulnessRow("MSFT", "Q1", "A3", 0.6, "A3", 2, 1),
    ]
    df = pd.DataFrame([vars(r) for r in rows])

    summary = summarize_faithfulness(df)

    assert "by_ticker" in summary
    assert "AAPL" in summary["by_ticker"]
    assert "MSFT" in summary["by_ticker"]
    assert summary["by_ticker"]["AAPL"]["n_queries"] == 2
    assert summary["by_ticker"]["MSFT"]["n_queries"] == 1


def test_summarize_mean_claims():
    """Verify calculation of mean claims per answer."""
    rows = [
        FaithfulnessRow("AAPL", "Q1", "A1", 0.8, "A1", 2, 1),
        FaithfulnessRow("AAPL", "Q2", "A2", 0.8, "A2", 4, 3),
        FaithfulnessRow("AAPL", "Q3", "A3", 0.8, "A3", 3, 2),
    ]
    df = pd.DataFrame([vars(r) for r in rows])

    summary = summarize_faithfulness(df)

    assert summary["mean_claims_per_answer"] == pytest.approx(3.0)


def test_summarize_supported_ratio():
    """Verify supported claims ratio calculation."""
    rows = [
        FaithfulnessRow("AAPL", "Q1", "A1", 0.5, "A1", 4, 2),
        FaithfulnessRow("AAPL", "Q2", "A2", 0.75, "A2", 4, 3),
    ]
    df = pd.DataFrame([vars(r) for r in rows])

    summary = summarize_faithfulness(df)

    expected_ratio = (2 + 3) / (4 + 4)
    assert summary["mean_supported_ratio"] == pytest.approx(expected_ratio, rel=0.01)


def test_summarize_faithfulness_perfect_scores():
    """Verify handling of perfect faithfulness scores."""
    rows = [
        FaithfulnessRow("AAPL", "Q1", "A1", 1.0, "A1", 1, 1),
        FaithfulnessRow("AAPL", "Q2", "A2", 1.0, "A2", 1, 1),
    ]
    df = pd.DataFrame([vars(r) for r in rows])

    summary = summarize_faithfulness(df)

    assert summary["mean_faithfulness"] == 1.0
    assert summary["median_faithfulness"] == 1.0
    assert summary["max_faithfulness"] == 1.0


def test_summarize_zero_faithfulness():
    """Verify handling of zero faithfulness scores."""
    rows = [
        FaithfulnessRow("AAPL", "Q1", "A1", 0.0, "A1", 3, 0),
        FaithfulnessRow("AAPL", "Q2", "A2", 0.0, "A2", 2, 0),
    ]
    df = pd.DataFrame([vars(r) for r in rows])

    summary = summarize_faithfulness(df)

    assert summary["mean_faithfulness"] == 0.0
    assert summary["median_faithfulness"] == 0.0
    assert summary["mean_supported_ratio"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
