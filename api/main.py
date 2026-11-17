"""FastAPI service exposing FinSight's company health scoring and RAG query endpoints."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import get_settings
from ingest.edgar import EdgarClient
from score.scorer import FinancialHealthScorer, LlamaScorer
from store.chroma import FilingStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing FinSight services (Chroma store, EDGAR client, LLM scorer)...")
    state["edgar_client"] = EdgarClient()
    state["store"] = FilingStore()
    state["llm"] = LlamaScorer()
    state["scorer"] = FinancialHealthScorer(
        store=state["store"], edgar_client=state["edgar_client"], llm=state["llm"]
    )
    yield
    state["edgar_client"].close()


app = FastAPI(
    title="FinSight",
    description="Company financial health scoring from SEC EDGAR filings via RAG + fine-tuned Llama.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScoreBreakdownResponse(BaseModel):
    revenue_trend: float
    risk_language: float
    executive_changes: float
    filing_sentiment: float


class ScoreResponse(BaseModel):
    ticker: str
    score: float
    breakdown: ScoreBreakdownResponse
    rationale: str
    filings_used: list[str]
    generated_at: str
    model: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question about a company's filings")
    ticker: str | None = Field(None, description="Restrict retrieval to this ticker")
    form_types: list[str] | None = Field(None, description='e.g. ["10-K"] or ["8-K"]')
    n_results: int = Field(5, ge=1, le=25)
    use_hybrid: bool = Field(True, description="Use hybrid (semantic + BM25) retrieval instead of semantic-only")


class SourceChunk(BaseModel):
    text: str
    metadata: dict
    distance: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/score/{ticker}", response_model=ScoreResponse)
def get_score(
    ticker: str,
    refresh: bool = Query(False, description="Re-ingest filings from EDGAR even if already stored"),
    filing_limit: int = Query(8, ge=1, le=50),
):
    """Fetch (or reuse) a ticker's filings and return its 1-10 FinSight health score."""
    scorer: FinancialHealthScorer = state["scorer"]
    try:
        result = scorer.score(ticker, refresh=refresh, filing_limit=filing_limit)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to score %s", ticker)
        raise HTTPException(status_code=500, detail=f"Failed to score {ticker}: {exc}") from exc
    return result.to_dict()


@app.post("/query", response_model=QueryResponse)
def query_filings(request: QueryRequest):
    """RAG query: retrieve relevant filing chunks and synthesize an answer with the LLM.

    Supports hybrid retrieval (semantic + BM25 keyword search via reciprocal rank fusion)
    for improved recall. Results include citation indices [1], [2], etc. for grounding."""
    store: FilingStore = state["store"]
    llm: LlamaScorer = state["llm"]

    if request.use_hybrid:
        chunks = store.query_hybrid(
            request.question,
            ticker=request.ticker,
            form_types=request.form_types,
            n_results=request.n_results,
        )
    else:
        chunks = store.query(
            request.question,
            ticker=request.ticker,
            form_types=request.form_types,
            n_results=request.n_results,
        )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No matching filing chunks found. Try /score/{ticker} first to ingest filings.",
        )

    context = "\n\n".join(
        f"[{c.citation_index}] {c.metadata['ticker']} {c.metadata['form_type']} {c.metadata['filing_date']}: {c.text}"
        for c in chunks
    )
    prompt = (
        "You are FinSight, a financial analyst assistant. Answer the question using ONLY the "
        "filing excerpts below. If the excerpts don't contain the answer, say so. "
        "When citing sources, use the citation numbers [1], [2], etc. that are provided.\n\n"
        f"Question: {request.question}\n\nFiling excerpts:\n{context}\n\nAnswer:"
    )

    try:
        # Greedy decoding on a small model loops on one sentence; penalize repeats for free-text answers.
        answer = llm.generate(
            prompt, max_new_tokens=300, repetition_penalty=1.2, no_repeat_ngram_size=4
        ).strip()
    except Exception:
        logger.exception("LLM unavailable for /query; returning retrieved sources only")
        answer = (
            "LLM generation is unavailable right now, so no synthesized answer could be produced. "
            "See the retrieved filing excerpts below."
        )

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=[SourceChunk(text=c.text, metadata=c.metadata, distance=c.distance) for c in chunks],
    )


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
