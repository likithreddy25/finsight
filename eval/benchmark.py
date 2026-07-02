"""Benchmark FinSight's fine-tuned Llama scorer against GPT-4o on the same companies.

For each ticker, both models see identical retrieved filing context and computed
signals (revenue trend, risk language, executive changes, filing sentiment) so the
comparison isolates differences in the *scoring judgment*, not the retrieval step.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass

import pandas as pd
from openai import OpenAI

from config.settings import get_settings
from score.scorer import FinancialHealthScorer, ScoreBreakdown

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "JPM", "XOM", "WMT", "NVDA", "META",
]

GPT4O_MODEL = "gpt-4o"


@dataclass
class BenchmarkRow:
    ticker: str
    finsight_score: float
    gpt4o_score: float
    abs_diff: float
    finsight_rationale: str
    gpt4o_rationale: str


def score_with_gpt4o(client: OpenAI, ticker: str, breakdown: ScoreBreakdown, context: str) -> tuple[float, str]:
    prompt = f"""You are a financial analyst scoring public companies' financial health from 1 \
(severe distress) to 10 (excellent health), based on SEC filings.

Company: {ticker}

Computed signals (already extracted from filings):
- revenue_trend: {breakdown.revenue_trend:.2f}  (-1 = shrinking, 0 = flat, +1 = strong growth)
- risk_language: {breakdown.risk_language:.2f}  (0 = little risk language, 1 = heavy risk language)
- executive_changes: {breakdown.executive_changes:.2f}  (0 = stable leadership, 1 = high churn)
- filing_sentiment: {breakdown.filing_sentiment:.2f}  (-1 = negative tone, +1 = positive tone)

Relevant filing excerpts:
{context}

Respond with ONLY a JSON object of the form:
{{"score": <number 1-10, one decimal place>, "rationale": "<2-4 sentence explanation citing the signals above>"}}
"""
    response = client.chat.completions.create(
        model=GPT4O_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    payload = json.loads(response.choices[0].message.content)
    return float(payload["score"]), str(payload["rationale"])


def run_benchmark(tickers: list[str]) -> pd.DataFrame:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (required to score with GPT-4o)")

    openai_client = OpenAI(api_key=settings.openai_api_key)
    scorer = FinancialHealthScorer()

    rows: list[BenchmarkRow] = []
    for ticker in tickers:
        logger.info("Scoring %s...", ticker)
        try:
            finsight_result = scorer.score(ticker)
            filings = scorer.edgar_client.get_filings(ticker)
            breakdown, context_chunks = scorer.compute_signals(ticker, filings)
            context = scorer.build_context(context_chunks)
            gpt4o_score, gpt4o_rationale = score_with_gpt4o(openai_client, ticker, breakdown, context)
        except Exception:
            logger.exception("Skipping %s due to an error", ticker)
            continue

        rows.append(
            BenchmarkRow(
                ticker=ticker,
                finsight_score=finsight_result.score,
                gpt4o_score=gpt4o_score,
                abs_diff=round(abs(finsight_result.score - gpt4o_score), 2),
                finsight_rationale=finsight_result.rationale,
                gpt4o_rationale=gpt4o_rationale,
            )
        )

    return pd.DataFrame(asdict(r) for r in rows)


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    return {
        "n": len(df),
        "mean_abs_diff": round(df["abs_diff"].mean(), 3),
        "max_abs_diff": round(df["abs_diff"].max(), 3),
        "correlation": round(df["finsight_score"].corr(df["gpt4o_score"]), 3) if len(df) > 1 else None,
        "finsight_mean": round(df["finsight_score"].mean(), 2),
        "gpt4o_mean": round(df["gpt4o_score"].mean(), 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FinSight vs GPT-4o on company health scores.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--output", default="eval/benchmark_results.csv")
    args = parser.parse_args()

    results_df = run_benchmark(args.tickers)
    results_df.to_csv(args.output, index=False)

    print(results_df.to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summarize(results_df), indent=2))
    print(f"\nSaved results to {args.output}")
