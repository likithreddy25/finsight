"""Tool-calling agent that autonomously orchestrates FinSight's EDGAR ingestion, ChromaDB
retrieval, and scoring pipeline to answer open-ended financial-health questions.

Unlike score/scorer.py's fixed pipeline (fetch -> compute signals -> retrieve -> score),
this agent decides for itself, turn by turn, which tools to call and in what order, based
on the user's actual question -- e.g. "compare AAPL and MSFT" triggers two independent
get_health_score calls plus a synthesis step; "any red flags in TSLA's latest 10-K" triggers
a targeted query_filings call instead of a full score. Every tool call is recorded in a
trace so the final answer is auditable back to the real filings it came from.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from openai import OpenAI

from config.settings import get_settings
from ingest.edgar import EdgarClient
from score.scorer import FinancialHealthScorer, LlamaScorer
from store.chroma import FilingStore

logger = logging.getLogger(__name__)

AGENT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are the FinSight Analyst Agent, a research assistant that investigates \
public companies using real SEC EDGAR filings via the tools available to you. You do not \
have reliable built-in knowledge of a company's current financials -- you must call tools to \
get real data before making any claim about a company's health, risk, or performance.

Guidelines:
- Break multi-company or multi-part questions into separate tool calls.
- Prefer get_health_score for "how healthy / how risky is X" questions.
- Prefer query_filings for specific, narrow questions ("what did X say about Y").
- Prefer get_revenue_facts when the question is purely about reported revenue figures.
- Always cite which filings/tickers your answer is grounded in.
- If a tool call fails or returns no data, say so plainly rather than guessing.
- Keep the final answer concise (3-6 sentences per company) and analyst-toned, not hype.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_health_score",
            "description": (
                "Fetch a company's real 8-K/10-K filings from SEC EDGAR (ingesting them if "
                "not already stored) and compute FinSight's 1-10 financial health score, with "
                "a breakdown of revenue trend, risk language, executive turnover, and filing "
                "sentiment signals."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker, e.g. AAPL"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_filings",
            "description": (
                "Semantic search over a company's stored SEC filing text for a specific "
                "question (e.g. supply chain risk, litigation, guidance). Returns the most "
                "relevant filing excerpts with source metadata. Call get_health_score first "
                "for a ticker if you're not sure its filings are already ingested."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "ticker": {"type": "string", "description": "Restrict to this ticker"},
                    "form_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["10-K", "8-K"]},
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_revenue_facts",
            "description": (
                "Fetch a company's raw XBRL-reported annual revenue figures (most recent "
                "fiscal years) directly from SEC's structured company-facts API."
            ),
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
]


@dataclass
class ToolCallRecord:
    tool: str
    arguments: dict
    result_summary: str


@dataclass
class AgentResult:
    query: str
    answer: str
    trace: list[ToolCallRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "trace": [
                {"tool": t.tool, "arguments": t.arguments, "result_summary": t.result_summary}
                for t in self.trace
            ],
        }


class FinSightAgent:
    """Tool-calling loop over FinSight's existing scoring/retrieval pipeline.

    This does not re-implement scoring or retrieval -- it wraps FinancialHealthScorer,
    FilingStore, and EdgarClient as callable tools and lets the LLM decide, turn by turn,
    which ones to invoke and how to synthesize the results into a final answer.
    """

    def __init__(
        self,
        store: FilingStore | None = None,
        edgar_client: EdgarClient | None = None,
        scorer: FinancialHealthScorer | None = None,
        openai_api_key: str | None = None,
        model: str = AGENT_MODEL,
        max_turns: int = 6,
    ):
        settings = get_settings()
        api_key = openai_api_key or settings.openai_api_key
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (required for the agent)")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_turns = max_turns

        self.edgar_client = edgar_client or EdgarClient()
        self.store = store or FilingStore()
        self.scorer = scorer or FinancialHealthScorer(
            store=self.store, edgar_client=self.edgar_client, llm=LlamaScorer()
        )

    # ---- tool implementations (thin wrappers over the existing pipeline) ----

    def _tool_get_health_score(self, ticker: str) -> dict:
        result = self.scorer.score(ticker)
        return result.to_dict()

    def _tool_query_filings(
        self, question: str, ticker: str | None = None, form_types: list[str] | None = None
    ) -> dict:
        chunks = self.store.query(question, ticker=ticker, form_types=form_types, n_results=6)
        if not chunks:
            return {
                "results": [],
                "note": "No stored filings matched -- try get_health_score first to ingest.",
            }
        return {
            "results": [
                {
                    "ticker": c.metadata.get("ticker"),
                    "form_type": c.metadata.get("form_type"),
                    "filing_date": c.metadata.get("filing_date"),
                    "text": c.text[:800],
                }
                for c in chunks
            ]
        }

    def _tool_get_revenue_facts(self, ticker: str) -> dict:
        facts = self.edgar_client.get_company_facts(ticker)
        gaap = facts.get("facts", {}).get("us-gaap", {})
        for tag in (
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
        ):
            entries = gaap.get(tag, {}).get("units", {}).get("USD", [])
            annual = sorted(
                (e for e in entries if e.get("form") == "10-K" and e.get("fp") == "FY" and e.get("val")),
                key=lambda e: e.get("end", ""),
            )
            if annual:
                return {"tag": tag, "recent_annual_revenue": annual[-3:]}
        return {"note": f"No annual revenue tag found for {ticker.upper()}"}

    def _dispatch(self, name: str, arguments: dict) -> dict:
        if name == "get_health_score":
            return self._tool_get_health_score(**arguments)
        if name == "query_filings":
            return self._tool_query_filings(**arguments)
        if name == "get_revenue_facts":
            return self._tool_get_revenue_facts(**arguments)
        raise ValueError(f"Unknown tool: {name}")

    # ---- main agent loop ----

    def run(self, query: str) -> AgentResult:
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        trace: list[ToolCallRecord] = []

        for _ in range(self.max_turns):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
            )
            message = response.choices[0].message

            if not message.tool_calls:
                return AgentResult(query=query, answer=message.content or "", trace=trace)

            messages.append(message.model_dump(exclude_unset=True))

            for call in message.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                try:
                    result = self._dispatch(call.function.name, args)
                    result_json = json.dumps(result, default=str)[:4000]
                except Exception as exc:
                    logger.exception("Tool %s failed", call.function.name)
                    result_json = json.dumps({"error": str(exc)})

                trace.append(
                    ToolCallRecord(tool=call.function.name, arguments=args, result_summary=result_json[:300])
                )
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result_json})

        return AgentResult(
            query=query, answer="Reached max tool-call turns without a final answer.", trace=trace
        )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ask the FinSight analyst agent a question.")
    parser.add_argument("query")
    args = parser.parse_args()

    agent = FinSightAgent()
    result = agent.run(args.query)
    print(json.dumps(result.to_dict(), indent=2))
