"""Faithfulness eval: measure whether QA answers are grounded in retrieved context.

For a fixed set of test questions per ticker, this runs the agent to get answers,
then uses a fact-checker to validate whether each claim in the answer is actually
supported by the retrieved chunks that were used to generate it.

Output: CSV with per-question faithfulness scores, and aggregate statistics.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from agent.analyst import FinSightAgent
from config.settings import get_settings
from score.critique import FactualClaimChecker
from store.chroma import FilingStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_QUERIES = {
    "AAPL": [
        "What is Apple's recent revenue trend and how does it compare to prior years?",
        "Describe any risk factors or operational challenges mentioned in the latest filings.",
        "What information is available about Apple's executive leadership changes?",
    ],
    "MSFT": [
        "How is Microsoft's financial health according to recent SEC filings?",
        "What revenue growth has Microsoft achieved recently?",
        "Are there any supply chain or operational risks disclosed?",
    ],
    "TSLA": [
        "Describe Tesla's current financial position and profitability trends.",
        "What risks does Tesla face according to its recent filings?",
        "How has Tesla's leadership team changed in recent filings?",
    ],
}


@dataclass
class FaithfulnessRow:
    ticker: str
    query: str
    answer: str
    faithfulness_score: float
    grounded_answer: str
    num_claims: int
    num_supported: int
    error: str = ""


def evaluate_single_query(
    agent: FinSightAgent | None,
    ticker: str,
    query: str,
    checker: FactualClaimChecker | None,
) -> FaithfulnessRow:
    """Run a single test query and evaluate faithfulness of the answer.

    Args:
        agent: The FinSight agent to generate answers
        ticker: Company ticker
        query: Test question
        checker: Fact-checker for validating claims

    Returns:
        FaithfulnessRow with evaluation results"""

    try:
        if agent is None:
            return FaithfulnessRow(
                ticker=ticker,
                query=query,
                answer="",
                faithfulness_score=0.0,
                grounded_answer="",
                num_claims=0,
                num_supported=0,
                error="Agent not available (OPENAI_API_KEY not set)",
            )

        result = agent.run(f"{query} Company: {ticker}")
        answer = result.answer

        if checker is None:
            return FaithfulnessRow(
                ticker=ticker,
                query=query,
                answer=answer,
                faithfulness_score=0.0,
                grounded_answer=answer,
                num_claims=0,
                num_supported=0,
                error="Fact-checker not available (OPENAI_API_KEY not set)",
            )

        critique_result = checker.critique_rationale(answer, [])
        num_supported = sum(1 for c in critique_result.claims if c.is_supported)
        faithfulness = (
            critique_result.faithfulness_score if critique_result.claims else 0.0
        )

        return FaithfulnessRow(
            ticker=ticker,
            query=query,
            answer=answer,
            faithfulness_score=faithfulness,
            grounded_answer=critique_result.grounded_rationale,
            num_claims=len(critique_result.claims),
            num_supported=num_supported,
        )

    except Exception as e:
        logger.exception("Error evaluating query for %s", ticker)
        return FaithfulnessRow(
            ticker=ticker,
            query=query,
            answer="",
            faithfulness_score=0.0,
            grounded_answer="",
            num_claims=0,
            num_supported=0,
            error=str(e),
        )


def run_faithfulness_eval(
    tickers: list[str], queries_per_ticker: dict[str, list[str]] | None = None
) -> pd.DataFrame:
    """Run faithfulness evaluation on a test set of queries.

    Args:
        tickers: List of tickers to evaluate
        queries_per_ticker: Custom queries per ticker (defaults to TEST_QUERIES)

    Returns:
        DataFrame with one row per (ticker, query) pair"""

    queries = queries_per_ticker or TEST_QUERIES
    settings = get_settings()

    try:
        agent = FinSightAgent()
    except RuntimeError:
        logger.warning("OPENAI_API_KEY not set; agent will be unavailable")
        agent = None

    try:
        checker = FactualClaimChecker()
    except RuntimeError:
        logger.warning("OPENAI_API_KEY not set; fact-checker will be unavailable")
        checker = None

    rows: list[FaithfulnessRow] = []

    for ticker in tickers:
        if ticker not in queries:
            logger.info("No test queries defined for %s; skipping", ticker)
            continue

        for query in queries[ticker]:
            logger.info("Evaluating %s: %s", ticker, query[:60])
            row = evaluate_single_query(agent, ticker, query, checker)
            rows.append(row)

    return pd.DataFrame(asdict(r) for r in rows)


def summarize_faithfulness(df: pd.DataFrame) -> dict:
    """Compute aggregate faithfulness statistics."""
    if df.empty:
        return {"n_queries": 0, "error": "No results"}

    errors = df[df["error"] != ""]
    valid_rows = df[df["error"] == ""]

    if valid_rows.empty:
        return {
            "n_queries": len(df),
            "n_errors": len(errors),
            "error_message": "All queries failed",
        }

    return {
        "n_queries": len(df),
        "n_errors": len(errors),
        "mean_faithfulness": round(valid_rows["faithfulness_score"].mean(), 3),
        "median_faithfulness": round(valid_rows["faithfulness_score"].median(), 3),
        "min_faithfulness": round(valid_rows["faithfulness_score"].min(), 3),
        "max_faithfulness": round(valid_rows["faithfulness_score"].max(), 3),
        "mean_claims_per_answer": round(valid_rows["num_claims"].mean(), 1),
        "mean_supported_ratio": round(
            (valid_rows["num_supported"].sum() / valid_rows["num_claims"].sum())
            if valid_rows["num_claims"].sum() > 0
            else 0,
            3,
        ),
        "by_ticker": {
            ticker: {
                "n_queries": len(ticker_rows),
                "mean_faithfulness": round(
                    ticker_rows[ticker_rows["error"] == ""]["faithfulness_score"].mean(), 3
                )
                if len(ticker_rows[ticker_rows["error"] == ""]) > 0
                else 0,
            }
            for ticker, ticker_rows in valid_rows.groupby("ticker")
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate faithfulness of FinSight agent answers to test queries."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(TEST_QUERIES.keys()),
        help="Tickers to evaluate (must have test queries defined)",
    )
    parser.add_argument(
        "--output",
        default="eval/faithfulness_results.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--test-queries",
        help="JSON file with custom test queries per ticker",
    )
    args = parser.parse_args()

    custom_queries = None
    if args.test_queries:
        with open(args.test_queries) as f:
            custom_queries = json.load(f)

    results_df = run_faithfulness_eval(args.tickers, queries_per_ticker=custom_queries)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)

    print("\nFaithfulness Evaluation Results:")
    print("=" * 80)
    print(results_df[["ticker", "query", "faithfulness_score", "num_supported", "num_claims"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("Summary Statistics:")
    print(json.dumps(summarize_faithfulness(results_df), indent=2))
    print(f"\nDetailed results saved to {args.output}")
