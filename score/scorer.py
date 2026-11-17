"""RAG + fine-tuned-LLM pipeline that turns SEC filings into a 1-10 company health score.

Pipeline:
1. Ingest (or reuse already-stored) 8-K/10-K filings for the ticker.
2. Compute four quantitative signals: revenue trend (XBRL), risk language density,
   executive-change density, and filing sentiment (lexicon heuristics over retrieved chunks).
3. Retrieve the most relevant chunks via ChromaDB and hand signals + context to a
   fine-tuned Llama LoRA model, which returns a final 1-10 score and rationale.
4. If the model is unavailable or returns something unparsable, fall back to a
   transparent weighted-heuristic score computed from the same four signals.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone

from config.settings import get_settings
from ingest.edgar import EdgarClient, Filing
from store.chroma import FilingStore, RetrievedChunk

logger = logging.getLogger(__name__)

REVENUE_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
]

RISK_KEYWORDS = [
    "material weakness", "going concern", "litigation", "impairment", "default",
    "restatement", "investigation", "non-compliance", "covenant violation",
    "adverse", "uncertainty", "volatility", "downturn", "recall", "breach",
    "restructuring", "layoff", "workforce reduction", "goodwill impairment",
]

POSITIVE_WORDS = [
    "growth", "increase", "improved", "strong", "record", "profitable",
    "expansion", "outperform", "robust", "beat expectations",
]

NEGATIVE_WORDS = [
    "decline", "decrease", "loss", "weak", "shortfall", "miss",
    "downturn", "impair", "reduced", "underperform",
]


@dataclass
class ScoreBreakdown:
    revenue_trend: float       # -1..1, positive = growing
    risk_language: float       # 0..1, higher = more risk language
    executive_changes: float   # 0..1, higher = more leadership churn
    filing_sentiment: float    # -1..1, positive = more positive tone


@dataclass
class HealthScore:
    ticker: str
    score: float               # 1..10
    breakdown: ScoreBreakdown
    rationale: str
    filings_used: list[str]
    generated_at: str
    model: str

    def to_dict(self) -> dict:
        return {**asdict(self), "breakdown": asdict(self.breakdown)}


class LlamaScorer:
    """Lazy-loading wrapper around the base Llama model + fine-tuned LoRA adapter."""

    def __init__(
        self,
        base_model_id: str | None = None,
        adapter_id: str | None = None,
        device: str | None = None,
        hf_token: str | None = None,
    ):
        settings = get_settings()
        self.base_model_id = base_model_id or settings.base_model_id
        self.adapter_id = adapter_id or settings.lora_adapter_id
        self.device = device or settings.model_device
        self.hf_token = hf_token or settings.hf_token
        self._tokenizer = None
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading base model %s + LoRA adapter %s", self.base_model_id, self.adapter_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, token=self.hf_token)
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            token=self.hf_token,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        )
        self._model = PeftModel.from_pretrained(base, self.adapter_id, token=self.hf_token)
        self._model.to(self.device)
        self._model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 400) -> str:
        self._load()
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True)


class FinancialHealthScorer:
    """Orchestrates ingestion, retrieval, signal extraction, and LLM scoring for a ticker."""

    def __init__(
        self,
        store: FilingStore | None = None,
        edgar_client: EdgarClient | None = None,
        llm: LlamaScorer | None = None,
    ):
        self.store = store or FilingStore()
        self.edgar_client = edgar_client or EdgarClient()
        self.llm = llm or LlamaScorer()

    def score(self, ticker: str, refresh: bool = False, filing_limit: int = 8) -> HealthScore:
        ticker = ticker.upper()
        filings = self.edgar_client.get_filings(ticker, limit=filing_limit)
        if not filings:
            raise ValueError(f"No 8-K/10-K filings found for {ticker!r}")

        if refresh or not self._has_stored_chunks(ticker):
            filings = self.edgar_client.get_filings_with_text(ticker, limit=filing_limit)
            self.store.add_filings(filings)

        breakdown, context_chunks = self.compute_signals(ticker, filings)
        context = self.build_context(context_chunks)
        prompt = self._build_prompt(ticker, breakdown, context)

        model_name = f"{self.llm.base_model_id} + {self.llm.adapter_id}"
        try:
            # Small models often ignore "respond with JSON"; start the object for them.
            prefill = '{"score": '
            raw_output = prefill + self.llm.generate(prompt + prefill)
            score_value, rationale = self._parse_llm_output(raw_output, breakdown)
            if rationale.startswith("Heuristic fallback"):
                model_name = "heuristic-fallback (LLM output unparsable)"
        except Exception:
            logger.exception("LLM scoring failed for %s; falling back to heuristic score", ticker)
            score_value, rationale = self._heuristic_score(breakdown)
            model_name = "heuristic-fallback (LLM unavailable)"

        return HealthScore(
            ticker=ticker,
            score=score_value,
            breakdown=breakdown,
            rationale=rationale,
            filings_used=[
                f"{f.form_type} {f.filing_date.isoformat()} ({f.accession_number})" for f in filings
            ],
            generated_at=datetime.now(timezone.utc).isoformat(),
            model=model_name,
        )

    def _has_stored_chunks(self, ticker: str) -> bool:
        return len(self.store.query("filing", ticker=ticker, n_results=1)) > 0

    def compute_signals(
        self, ticker: str, filings: list[Filing]
    ) -> tuple[ScoreBreakdown, list[RetrievedChunk]]:
        risk_chunks = self.store.query(
            "risk factors litigation material weakness uncertainty",
            ticker=ticker,
            form_types=["10-K"],
            n_results=10,
        )
        exec_chunks = self.store.query(
            "executive officer resignation appointment departure board of directors",
            ticker=ticker,
            form_types=["8-K"],
            n_results=10,
        )
        outlook_chunks = self.store.query(
            "results of operations business outlook performance guidance",
            ticker=ticker,
            n_results=10,
        )

        breakdown = ScoreBreakdown(
            revenue_trend=self._revenue_trend(ticker),
            risk_language=self._risk_language_score(risk_chunks),
            executive_changes=self._executive_change_score(filings),
            filing_sentiment=self._sentiment_score(outlook_chunks),
        )
        context_chunks = (risk_chunks + exec_chunks + outlook_chunks)[:12]
        return breakdown, context_chunks

    def _revenue_trend(self, ticker: str) -> float:
        try:
            facts = self.edgar_client.get_company_facts(ticker)
        except Exception:
            logger.warning("Could not fetch XBRL company facts for %s", ticker)
            return 0.0

        gaap_facts = facts.get("facts", {}).get("us-gaap", {})
        for tag in REVENUE_TAGS:
            entries = gaap_facts.get(tag, {}).get("units", {}).get("USD", [])
            annual = sorted(
                (e for e in entries if self._is_annual_10k_entry(e)),
                key=lambda e: e["end"],
            )
            if len(annual) >= 2:
                prev_val, latest_val = annual[-2]["val"], annual[-1]["val"]
                if prev_val:
                    growth = (latest_val - prev_val) / abs(prev_val)
                    return max(-1.0, min(1.0, growth / 0.2))
        return 0.0

    @staticmethod
    def _is_annual_10k_entry(entry: dict) -> bool:
        """True if an XBRL fact represents a full fiscal year (not a quarterly comparative
        reported inside the same 10-K). companyfacts tags every context in a filing with
        fp="FY", so duration must also be checked to exclude quarterly figures."""
        if entry.get("form") != "10-K" or entry.get("fp") != "FY" or not entry.get("val"):
            return False
        try:
            start = date.fromisoformat(entry["start"])
            end = date.fromisoformat(entry["end"])
        except (KeyError, ValueError):
            return False
        return 330 <= (end - start).days <= 380

    @staticmethod
    def _risk_language_score(chunks: list[RetrievedChunk]) -> float:
        if not chunks:
            return 0.0
        combined = " ".join(c.text.lower() for c in chunks)
        word_count = max(len(combined.split()), 1)
        hits = sum(combined.count(kw) for kw in RISK_KEYWORDS)
        density_per_1k_words = hits / word_count * 1000
        return max(0.0, min(1.0, density_per_1k_words / 8.0))

    @staticmethod
    def _executive_change_score(filings: list[Filing]) -> float:
        """Fraction of recent 8-Ks reporting Item 5.02 (departure/election/appointment of
        officers or directors). Uses SEC's structured item codes rather than keyword
        matching, since officer/director titles appear in most 8-K boilerplate regardless
        of which item was actually triggered."""
        eight_ks = [f for f in filings if f.form_type == "8-K"]
        if not eight_ks:
            return 0.0
        exec_change_count = sum(1 for f in eight_ks if "5.02" in f.items.split(","))
        return max(0.0, min(1.0, exec_change_count / len(eight_ks)))

    @staticmethod
    def _sentiment_score(chunks: list[RetrievedChunk]) -> float:
        if not chunks:
            return 0.0
        combined = " ".join(c.text.lower() for c in chunks)
        pos = sum(combined.count(w) for w in POSITIVE_WORDS)
        neg = sum(combined.count(w) for w in NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            return 0.0
        return max(-1.0, min(1.0, (pos - neg) / total))

    @staticmethod
    def build_context(chunks: list[RetrievedChunk], max_chars: int = 4000) -> str:
        parts = []
        used = 0
        for c in chunks:
            snippet = f"[{c.metadata['form_type']} {c.metadata['filing_date']}] {c.text}"
            if used + len(snippet) > max_chars:
                break
            parts.append(snippet)
            used += len(snippet)
        return "\n\n".join(parts)

    @staticmethod
    def _build_prompt(ticker: str, breakdown: ScoreBreakdown, context: str) -> str:
        return f"""You are FinSight, a financial analyst model that scores public companies' \
financial health from 1 (severe distress) to 10 (excellent health), based on SEC filings.

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

    @staticmethod
    def _parse_llm_output(raw_output: str, breakdown: ScoreBreakdown) -> tuple[float, str]:
        start = raw_output.find("{")
        match = start != -1
        if match:
            try:
                # Models often keep generating after the first object; read only the first.
                payload, _ = json.JSONDecoder().raw_decode(raw_output[start:])
                score_value = max(1.0, min(10.0, float(payload["score"])))
                rationale = str(payload["rationale"])
                return round(score_value, 1), rationale
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                logger.warning("Could not parse LLM output as JSON: %r", raw_output)
        else:
            logger.warning("LLM output contained no JSON object: %r", raw_output[:800])
        return FinancialHealthScorer._heuristic_score(breakdown)

    @staticmethod
    def _heuristic_score(breakdown: ScoreBreakdown) -> tuple[float, str]:
        base = 5.5
        base += breakdown.revenue_trend * 2.0
        base -= breakdown.risk_language * 2.0
        base -= breakdown.executive_changes * 1.5
        base += breakdown.filing_sentiment * 1.5
        score_value = round(max(1.0, min(10.0, base)), 1)
        rationale = (
            "Heuristic fallback score (LLM unavailable or output unparsable): "
            f"revenue_trend={breakdown.revenue_trend:.2f}, risk_language={breakdown.risk_language:.2f}, "
            f"executive_changes={breakdown.executive_changes:.2f}, filing_sentiment={breakdown.filing_sentiment:.2f}."
        )
        return score_value, rationale


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Compute a FinSight health score for a ticker.")
    parser.add_argument("ticker")
    parser.add_argument("--refresh", action="store_true", help="Re-ingest filings even if already stored")
    args = parser.parse_args()

    scorer = FinancialHealthScorer()
    result = scorer.score(args.ticker, refresh=args.refresh)
    print(json.dumps(result.to_dict(), indent=2))
