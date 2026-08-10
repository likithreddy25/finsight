"""Critique loop: validate that scored rationales are grounded in retrieved chunks."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from openai import OpenAI

from config.settings import get_settings
from store.chroma import RetrievedChunk

logger = logging.getLogger(__name__)

CRITIQUE_MODEL = "gpt-4o-mini"


@dataclass
class ClaimCheckResult:
    """Result of fact-checking a single claim."""

    claim: str
    is_supported: bool
    explanation: str
    supporting_chunk_indices: list[int]


@dataclass
class CritiqueResult:
    """Result of critiquing a full rationale."""

    original_rationale: str
    claims: list[ClaimCheckResult]
    grounded_rationale: str
    faithfulness_score: float


class FactualClaimChecker:
    """Validates claims in scored rationales against retrieved filing chunks."""

    def __init__(self, openai_api_key: str | None = None, model: str = CRITIQUE_MODEL):
        settings = get_settings()
        api_key = openai_api_key or settings.openai_api_key
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (required for fact-checking)")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def critique_rationale(
        self, rationale: str, chunks: list[RetrievedChunk]
    ) -> CritiqueResult:
        """Check whether each claim in a rationale is grounded in the retrieved chunks.

        Args:
            rationale: The text rationale to fact-check (e.g., from FinancialHealthScorer)
            chunks: The list of retrieved chunks that were used to generate the rationale

        Returns:
            CritiqueResult with original rationale, per-claim analysis, and grounded version"""
        context = self._build_context(chunks)
        prompt = self._build_critique_prompt(rationale, context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            raw_output = response.choices[0].message.content or ""
            result = self._parse_critique_response(raw_output, rationale, chunks)
        except Exception:
            logger.exception("Critique failed; returning original rationale unchecked")
            result = CritiqueResult(
                original_rationale=rationale,
                claims=[],
                grounded_rationale=rationale,
                faithfulness_score=0.0,
            )

        return result

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format chunks as numbered context for the critic to reference."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            snippet = f"[{i}] {chunk.metadata['form_type']} {chunk.metadata['filing_date']}: {chunk.text}"
            parts.append(snippet)
        return "\n\n".join(parts)

    @staticmethod
    def _build_critique_prompt(rationale: str, context: str) -> str:
        return f"""You are a fact-checker for a financial analysis system. Your job is to check
whether each claim in a generated rationale is actually supported by the provided filing excerpts.

Filing excerpts (numbered [1], [2], etc.):
{context}

Rationale to fact-check:
{rationale}

For each distinct factual claim in the rationale, determine whether it is supported by the
filing excerpts provided. A claim is "supported" if it is explicitly stated or clearly follows
from the excerpts. A claim is "unsupported" if it requires knowledge not in the excerpts.

Respond with ONLY a JSON object of this form:
{{
  "claims": [
    {{
      "claim": "<the exact claim from the rationale>",
      "is_supported": true/false,
      "explanation": "<brief explanation of why it is or isn't supported>",
      "supporting_chunk_indices": [<1-based indices of supporting chunks, e.g. [1, 3]>]
    }},
    ...
  ]
}}

Be precise: only mark claims as supported if the evidence is clearly in the excerpts."""

    @staticmethod
    def _parse_critique_response(
        raw_output: str, rationale: str, chunks: list[RetrievedChunk]
    ) -> CritiqueResult:
        """Parse the critic's JSON response and construct a CritiqueResult."""
        try:
            match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if not match:
                raise ValueError("No JSON found in response")

            payload = json.loads(match.group(0))
            claims_data = payload.get("claims", [])

            claims: list[ClaimCheckResult] = []
            supported_count = 0

            for claim_data in claims_data:
                claim = ClaimCheckResult(
                    claim=claim_data.get("claim", ""),
                    is_supported=claim_data.get("is_supported", False),
                    explanation=claim_data.get("explanation", ""),
                    supporting_chunk_indices=claim_data.get("supporting_chunk_indices", []),
                )
                claims.append(claim)
                if claim.is_supported:
                    supported_count += 1

            faithfulness_score = supported_count / len(claims) if claims else 0.0

            grounded_rationale = FactualClaimChecker._build_grounded_rationale(
                rationale, claims
            )

            return CritiqueResult(
                original_rationale=rationale,
                claims=claims,
                grounded_rationale=grounded_rationale,
                faithfulness_score=faithfulness_score,
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Could not parse critique response: %s", e)
            return CritiqueResult(
                original_rationale=rationale,
                claims=[],
                grounded_rationale=rationale,
                faithfulness_score=0.0,
            )

    @staticmethod
    def _build_grounded_rationale(rationale: str, claims: list[ClaimCheckResult]) -> str:
        """Construct a new rationale that only includes supported claims, adding
        'not found in retrieved filings' notices for unsupported ones."""
        if not claims:
            return rationale

        unsupported_claims = [c for c in claims if not c.is_supported]
        if not unsupported_claims:
            return rationale

        grounded = rationale
        for claim in unsupported_claims:
            grounded = grounded.replace(
                claim.claim, f"{claim.claim} [not grounded in retrieved filings]"
            )

        return grounded


if __name__ == "__main__":
    from store.chroma import FilingStore
    from ingest.edgar import EdgarClient

    checker = FactualClaimChecker()
    store = FilingStore()

    ticker = "AAPL"
    with EdgarClient() as client:
        filings = client.get_filings_with_text(ticker, limit=1)
    store.add_filings(filings)

    chunks = store.query(
        "revenue growth and risk factors", ticker=ticker, n_results=5
    )

    fake_rationale = "Revenue grew 50% YoY. The company face severe liquidity crisis. Strong market position."
    result = checker.critique_rationale(fake_rationale, chunks)

    print(f"\nOriginal: {result.original_rationale}")
    print(f"\nFaithfulness score: {result.faithfulness_score:.2%}")
    for claim in result.claims:
        status = "✓" if claim.is_supported else "✗"
        print(f"\n{status} {claim.claim}")
        print(f"   {claim.explanation}")
    print(f"\nGrounded: {result.grounded_rationale}")
