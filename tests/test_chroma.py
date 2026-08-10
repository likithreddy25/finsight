"""Tests for hybrid retrieval and citation grounding in store/chroma.py."""
from __future__ import annotations

import tempfile
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from ingest.edgar import Filing
from store.chroma import FilingStore, RetrievedChunk


@pytest.fixture
def temp_store():
    """Fixture: FilingStore with temporary ChromaDB directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield FilingStore(persist_dir=tmpdir)


def create_test_filing(
    ticker: str,
    accession: str,
    form_type: str,
    text: str,
) -> Filing:
    """Helper to create a Filing object for testing."""
    filing = MagicMock(spec=Filing)
    filing.ticker = ticker
    filing.accession_number = accession
    filing.form_type = form_type
    filing.cik = "0000000000"
    filing.filing_date = date(2024, 1, 15)
    filing.report_date = date(2023, 12, 31)
    filing.document_url = "http://example.com/filing"
    filing.items = ""
    filing.text = text
    return filing


def test_semantic_search_sets_citation_index(temp_store):
    """Verify semantic search results have sequential citation indices."""
    filing = create_test_filing(
        "TEST",
        "0000000001",
        "10-K",
        "Revenue growth this year was strong. Profitability improved. Margins expanded.",
    )
    temp_store.add_filing(filing)

    results = temp_store.query("revenue and profitability", ticker="TEST")

    assert len(results) > 0
    for i, chunk in enumerate(results):
        assert chunk.citation_index == i + 1
        assert isinstance(chunk.distance, float)
        assert chunk.metadata["ticker"] == "TEST"


def test_hybrid_search_combines_semantic_and_keyword(temp_store):
    """Verify hybrid search returns results combining both semantic and keyword matching."""
    filing = create_test_filing(
        "TEST",
        "0000000001",
        "10-K",
        """
        Risk factors section. The company faces material litigation risks related to supply chain
        disruptions. Key executives have experience in supply chain management. Supply chain
        resilience is critical. Suppliers operate globally across multiple jurisdictions.
        Key vendors located in Asia Pacific region. Risk mitigation involves supply chain
        diversification strategies.
        """,
    )
    temp_store.add_filing(filing)

    hybrid_results = temp_store.query_hybrid("supply chain risk", ticker="TEST")

    assert len(hybrid_results) > 0
    for chunk in hybrid_results:
        assert chunk.citation_index > 0
        assert "supply chain" in chunk.text.lower() or "risk" in chunk.text.lower()


def test_hybrid_search_vs_semantic_only(temp_store):
    """Verify hybrid search can surface results that semantic-only might miss."""
    filing = create_test_filing(
        "TEST",
        "0000000001",
        "10-K",
        """
        The company reported strong fiscal performance. Revenue increased by 15% YoY.
        Operating margins improved to 22% from 20% prior year. EBITDA grew substantially.
        Cash flow remained robust throughout the period. Working capital management was effective.
        The board approved a dividend increase of 5%.
        """,
    )
    temp_store.add_filing(filing)

    semantic_results = temp_store.query("dividend payout board", ticker="TEST")
    hybrid_results = temp_store.query_hybrid("dividend payout board", ticker="TEST")

    assert len(semantic_results) > 0
    assert len(hybrid_results) > 0
    assert len(hybrid_results) <= len(semantic_results) + 5


def test_citation_indices_are_sequential(temp_store):
    """Verify citation indices are always sequential starting from 1."""
    filing = create_test_filing(
        "TEST",
        "0000000001",
        "10-K",
        """First section talks about market trends and growth prospects.
        Second section discusses management team and organizational structure.
        Third section covers financial performance and key metrics.
        Fourth section details risk factors and mitigation strategies.""",
    )
    temp_store.add_filing(filing)

    hybrid_results = temp_store.query_hybrid("market growth team", ticker="TEST", n_results=4)

    citation_indices = [chunk.citation_index for chunk in hybrid_results]
    expected = list(range(1, len(hybrid_results) + 1))
    assert citation_indices == expected, f"Expected {expected}, got {citation_indices}"


def test_citation_grounding_multiple_filings(temp_store):
    """Verify citations properly track which chunks came from which filings."""
    filing1 = create_test_filing("TEST", "0000000001", "10-K", "Revenue 2024: $100M. Growth 15%.")
    filing2 = create_test_filing("TEST", "0000000002", "8-K", "Announced Q3 earnings: $25M revenue.")

    temp_store.add_filing(filing1)
    temp_store.add_filing(filing2)

    results = temp_store.query_hybrid("earnings revenue", ticker="TEST", n_results=5)

    assert len(results) > 0
    for chunk in results:
        assert chunk.citation_index > 0
        assert chunk.metadata["accession_number"] in ["0000000001", "0000000002"]
        assert chunk.metadata["ticker"] == "TEST"


@pytest.mark.parametrize(
    "query,expected_keywords",
    [
        ("litigation risk", ["litigation", "risk", "dispute"]),
        ("executive compensation", ["executive", "compensation", "salary"]),
        ("market share", ["market", "share", "percentage"]),
    ],
)
def test_hybrid_search_matches_query_terms(temp_store, query, expected_keywords):
    """Verify hybrid search returns chunks containing query-relevant keywords."""
    text = """
    The company faces litigation risks from multiple jurisdictions. Recent disputes with
    vendors have been resolved. Executive compensation includes salary and equity awards.
    Equity compensation aligns management with shareholder interests. Market share increased
    in key segments. Percentage growth exceeded industry benchmarks.
    """
    filing = create_test_filing("TEST", "0000000001", "10-K", text)
    temp_store.add_filing(filing)

    results = temp_store.query_hybrid(query, ticker="TEST")

    assert len(results) > 0
    combined_text = " ".join(r.text.lower() for r in results)
    for keyword in expected_keywords:
        assert keyword in combined_text


def test_empty_results_for_nonexistent_ticker(temp_store):
    """Verify empty results when querying for a ticker with no stored filings."""
    results = temp_store.query_hybrid("revenue growth", ticker="NONEXISTENT")
    assert results == []

    results_semantic = temp_store.query("revenue growth", ticker="NONEXISTENT")
    assert results_semantic == []


def test_form_type_filter_works_with_hybrid(temp_store):
    """Verify form_types parameter filters results correctly in hybrid search."""
    filing_10k = create_test_filing("TEST", "0000000001", "10-K", "Annual report content with risk disclosure.")
    filing_8k = create_test_filing("TEST", "0000000002", "8-K", "Current report of material event news.")

    temp_store.add_filing(filing_10k)
    temp_store.add_filing(filing_8k)

    results_10k = temp_store.query_hybrid("disclosure event", ticker="TEST", form_types=["10-K"])
    results_8k = temp_store.query_hybrid("disclosure event", ticker="TEST", form_types=["8-K"])

    assert len(results_10k) > 0
    assert len(results_8k) > 0

    for chunk in results_10k:
        assert chunk.metadata["form_type"] == "10-K"

    for chunk in results_8k:
        assert chunk.metadata["form_type"] == "8-K"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
