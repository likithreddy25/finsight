"""Tests for the factual claim checker / critique loop."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from score.critique import ClaimCheckResult, CritiqueResult, FactualClaimChecker
from store.chroma import RetrievedChunk


@pytest.fixture
def sample_chunks():
    """Fixture: sample retrieved chunks for testing."""
    chunks = [
        RetrievedChunk(
            text="Apple Inc. reported revenue of $383 billion in fiscal 2024, up 4% from $368 billion in 2023.",
            metadata={
                "ticker": "AAPL",
                "form_type": "10-K",
                "filing_date": "2024-10-25",
            },
            distance=0.1,
            citation_index=1,
        ),
        RetrievedChunk(
            text="The company faces supply chain risks related to manufacturing in Asia. Diversification efforts are ongoing.",
            metadata={
                "ticker": "AAPL",
                "form_type": "10-K",
                "filing_date": "2024-10-25",
            },
            distance=0.2,
            citation_index=2,
        ),
        RetrievedChunk(
            text="Operating margins remained stable at 30%. The company maintained strong cash positions with $161 billion in cash and equivalents.",
            metadata={
                "ticker": "AAPL",
                "form_type": "10-K",
                "filing_date": "2024-10-25",
            },
            distance=0.3,
            citation_index=3,
        ),
    ]
    return chunks


@pytest.fixture
def mock_openai_client(sample_chunks):
    """Fixture: mock OpenAI client with realistic responses."""
    def make_response(rationale: str):
        if "crisis" in rationale.lower():
            response_text = json.dumps({
                "claims": [
                    {
                        "claim": "Revenue grew 50% YoY",
                        "is_supported": False,
                        "explanation": "The filings show 4% revenue growth, not 50%.",
                        "supporting_chunk_indices": [],
                    },
                    {
                        "claim": "The company faces supply chain risks",
                        "is_supported": True,
                        "explanation": "Explicitly mentioned in the 10-K risk factors.",
                        "supporting_chunk_indices": [2],
                    },
                    {
                        "claim": "Company is in severe liquidity crisis",
                        "is_supported": False,
                        "explanation": "Filings show $161 billion in cash, contradicting a crisis claim.",
                        "supporting_chunk_indices": [],
                    },
                ]
            })
        else:
            response_text = json.dumps({
                "claims": [
                    {
                        "claim": "Strong cash position",
                        "is_supported": True,
                        "explanation": "Clearly stated in the filings: $161 billion in cash.",
                        "supporting_chunk_indices": [3],
                    },
                    {
                        "claim": "Stable operating margins",
                        "is_supported": True,
                        "explanation": "Filing shows 30% operating margins.",
                        "supporting_chunk_indices": [3],
                    },
                ]
            })

        message = MagicMock()
        message.content = response_text
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        return response

    client = MagicMock()
    client.chat.completions.create.side_effect = lambda **kwargs: make_response(
        kwargs["messages"][0]["content"]
    )
    return client


def test_critique_detects_false_claims(sample_chunks, mock_openai_client):
    """Verify that the critic catches fabricated claims."""
    checker = FactualClaimChecker(openai_api_key="sk-test-key")
    checker.client = mock_openai_client

    false_rationale = "Revenue grew 50% YoY. The company faces supply chain risks. Company is in severe liquidity crisis."
    result = checker.critique_rationale(false_rationale, sample_chunks)

    assert len(result.claims) == 3
    assert not result.claims[0].is_supported
    assert result.claims[1].is_supported
    assert not result.claims[2].is_supported
    assert result.faithfulness_score == pytest.approx(1 / 3)


def test_critique_confirms_supported_claims(sample_chunks, mock_openai_client):
    """Verify that the critic confirms factually accurate claims."""
    checker = FactualClaimChecker(openai_api_key="sk-test-key")
    checker.client = mock_openai_client

    true_rationale = "Strong cash position and stable operating margins."
    result = checker.critique_rationale(true_rationale, sample_chunks)

    assert len(result.claims) == 2
    assert all(c.is_supported for c in result.claims)
    assert result.faithfulness_score == 1.0


def test_grounded_rationale_adds_markers(sample_chunks, mock_openai_client):
    """Verify that unsupported claims are marked with [not grounded in retrieved filings]."""
    checker = FactualClaimChecker(openai_api_key="sk-test-key")
    checker.client = mock_openai_client

    false_rationale = "Revenue grew 50% YoY. The company faces supply chain risks. Company is in severe liquidity crisis."
    result = checker.critique_rationale(false_rationale, sample_chunks)

    assert "[not grounded in retrieved filings]" in result.grounded_rationale
    assert "Revenue grew 50% YoY" in result.grounded_rationale
    assert "Company is in severe liquidity crisis" in result.grounded_rationale


def test_supporting_chunk_indices_tracked(sample_chunks, mock_openai_client):
    """Verify that the checker tracks which chunks support each claim."""
    checker = FactualClaimChecker(openai_api_key="sk-test-key")
    checker.client = mock_openai_client

    rationale = "Company is in severe liquidity crisis but The company faces supply chain risks."
    result = checker.critique_rationale(rationale, sample_chunks)

    supply_chain_claims = [c for c in result.claims if "supply chain" in c.claim.lower()]
    assert len(supply_chain_claims) > 0
    assert supply_chain_claims[0].is_supported
    assert 2 in supply_chain_claims[0].supporting_chunk_indices


def test_context_building_includes_chunk_numbers(sample_chunks):
    """Verify that the context passed to the critic has numbered references [1], [2], etc."""
    checker = FactualClaimChecker(openai_api_key="sk-test-key")
    context = checker._build_context(sample_chunks)

    assert "[1]" in context
    assert "[2]" in context
    assert "[3]" in context
    assert "10-K" in context
    assert "revenue" in context.lower()


def test_critique_prompt_includes_instructions():
    """Verify that the critique prompt includes clear instructions for the critic."""
    prompt = FactualClaimChecker._build_critique_prompt("Test claim", "[1] Sample context")

    assert "fact-check" in prompt.lower()
    assert "supported" in prompt.lower()
    assert "JSON" in prompt
    assert "[1]" in prompt


def test_empty_chunks_list():
    """Verify handling when no chunks are provided."""
    checker = FactualClaimChecker.__new__(FactualClaimChecker)
    context = checker._build_context([])
    assert context == ""


def test_malformed_json_response_fallback(sample_chunks):
    """Verify graceful fallback when critic returns unparseable JSON."""
    bad_client = MagicMock()
    bad_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Not valid JSON at all"))]
    )

    checker = FactualClaimChecker(openai_api_key="sk-test-key")
    checker.client = bad_client

    result = checker.critique_rationale("Test rationale", sample_chunks)

    assert result.original_rationale == "Test rationale"
    assert result.claims == []
    assert result.grounded_rationale == "Test rationale"
    assert result.faithfulness_score == 0.0


def test_parse_multiple_claims():
    """Verify parsing of response with multiple claims."""
    response_json = json.dumps({
        "claims": [
            {
                "claim": "Claim 1",
                "is_supported": True,
                "explanation": "Supported",
                "supporting_chunk_indices": [1],
            },
            {
                "claim": "Claim 2",
                "is_supported": False,
                "explanation": "Not supported",
                "supporting_chunk_indices": [],
            },
            {
                "claim": "Claim 3",
                "is_supported": True,
                "explanation": "Supported",
                "supporting_chunk_indices": [2, 3],
            },
        ]
    })

    chunks = [
        RetrievedChunk(
            text="Sample 1",
            metadata={"form_type": "10-K", "filing_date": "2024-01-01"},
            distance=0.1,
            citation_index=1,
        ),
    ]

    result = FactualClaimChecker._parse_critique_response(response_json, "Test", chunks)

    assert len(result.claims) == 3
    assert result.claims[0].is_supported
    assert not result.claims[1].is_supported
    assert result.claims[2].is_supported
    assert result.faithfulness_score == pytest.approx(2 / 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
