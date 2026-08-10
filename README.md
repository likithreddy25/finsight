# FinSight

Company financial health scoring from real SEC EDGAR filings, using retrieval-augmented
generation (RAG) over 8-K/10-K text plus a fine-tuned Llama model, backed by quantitative
signals pulled directly from SEC's structured filing data.

**Not investment advice.** Scores are a research/engineering artifact, not a recommendation.

## How it works

```
ticker ──> ingest/edgar.py ──> store/chroma.py ──> score/scorer.py ──> api/main.py ──> dashboard/
           (SEC EDGAR REST      (chunk + embed in      (signals + RAG    (FastAPI:         (React +
            APIs, live data)     ChromaDB)              + fine-tuned      /score, /query)   TypeScript)
                                                         Llama LoRA)
```

1. **`ingest/edgar.py`** resolves a ticker to its SEC CIK, pulls recent 8-K/10-K filings from
   `data.sec.gov/submissions`, and downloads each filing's primary document from EDGAR's
   archives. It also fetches structured XBRL "company facts" (e.g. reported revenue) and each
   8-K's structured item codes (e.g. `5.02` = officer/director change) — used for signals
   instead of guessing from free text.
2. **`store/chroma.py`** chunks filing text and embeds it with `sentence-transformers`,
   persisting everything in a local ChromaDB collection tagged with ticker/form/date metadata.
3. **`score/scorer.py`** computes four signals per company:
   - **Revenue trend** — YoY growth from XBRL-reported annual revenue.
   - **Risk language** — keyword density in retrieved 10-K risk-factor chunks.
   - **Executive changes** — fraction of recent 8-Ks reporting Item 5.02.
   - **Filing sentiment** — lexicon-based tone over retrieved MD&A/outlook chunks.

   These signals, plus the most relevant retrieved chunks, are handed to a fine-tuned Llama
   model (`likilogan/finsight-llama-lora-v1`, a LoRA adapter over a Llama 3.1 base) which
   returns a final 1-10 score and rationale. If the model isn't available (no GPU, no HF
   token/gated access, etc.) the API transparently falls back to a weighted-heuristic score
   built from the same four signals, and says so in the response's `model` field.
4. **`api/main.py`** exposes `GET /score/{ticker}` and `POST /query` (free-text RAG Q&A over
   stored filings) via FastAPI.
5. **`dashboard/`** is a React + TypeScript UI (Vite, Recharts) that shows the score, a
   breakdown chart of the four signals, and the filings that were used.
6. **`eval/benchmark.py`** runs the same signals/context through GPT-4o for a set of tickers
   and compares its scores against FinSight's, to sanity-check the fine-tuned model.

## Requirements

- Python 3.11
- Node.js 18+ (for the dashboard)
- A Hugging Face token with access to the base Llama model and the LoRA adapter, if you want
  real (non-heuristic) scoring — see [Model access](#model-access).

## Setup (local, no Docker)

```bash
# Backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in SEC_USER_AGENT at minimum

uvicorn api.main:app --reload   # http://localhost:8000

# Dashboard (separate terminal)
cd dashboard
npm install
cp .env.example .env   # VITE_API_BASE_URL, defaults to http://localhost:8000
npm run dev              # http://localhost:5173
```

`SEC_USER_AGENT` must identify you (e.g. `"Your Name your-email@example.com"`) per
[SEC's fair-access policy](https://www.sec.gov/os/webmaster-faq#developers) — requests without
one are more likely to be rate-limited or blocked.

## Setup (Docker)

```bash
cp .env.example .env   # fill in SEC_USER_AGENT, HF_TOKEN, etc.
docker compose up --build
```

- API: http://localhost:8000
- Dashboard: http://localhost:5173

ChromaDB persists to a named volume (`chroma_data`) so ingested filings survive container
restarts.

## Model access

The scorer uses a base Llama model (`BASE_MODEL_ID`, default
`meta-llama/Llama-3.1-8B-Instruct`) plus a LoRA adapter (`LORA_ADAPTER_ID`, default
`likilogan/finsight-llama-lora-v1`) loaded via `transformers` + `peft`. Both typically require
an `HF_TOKEN` with access granted to the base model's gated repo. Without it — or without
`torch`/`transformers`/`peft` installed, or without a GPU — `score/scorer.py` catches the
failure and returns a heuristic score instead, so the API stays usable in dev environments.
Set `MODEL_DEVICE=cuda` (or `mps` on Apple Silicon) once you have real hardware for it.

## API

### `GET /score/{ticker}?refresh=false&filing_limit=8`

Returns the 1-10 health score, per-signal breakdown, rationale, filings used, and which model
produced it (fine-tuned LLM vs. heuristic fallback). Set `refresh=true` to re-ingest the
ticker's filings from EDGAR even if already stored.

### `POST /query`

```json
{
  "question": "What did the company say about supply chain risk?",
  "ticker": "AAPL",
  "form_types": ["10-K"],
  "n_results": 5,
  "use_hybrid": true
}
```

Retrieves the most relevant stored filing chunks and asks the LLM to answer using only that
context (falls back to returning the raw chunks if the LLM is unavailable).

**Hybrid retrieval** (`use_hybrid: true`, default): combines semantic search (via sentence-transformers embeddings
in ChromaDB) with BM25 keyword matching. Results are merged using reciprocal rank fusion,
giving equal weight to both signals. This improves recall for questions where keyword
matches are important (e.g., "Item 8-K 5.02" for executive changes). Results include
inline citation indices `[1]`, `[2]` etc. for grounding.

**Semantic-only** (`use_hybrid: false`): pure embedding-based search via ChromaDB, faster
and sufficient for questions where semantic relevance dominates.

## Evaluation

```bash
export OPENAI_API_KEY=sk-...
python -m eval.benchmark --tickers AAPL MSFT GOOGL AMZN TSLA JPM XOM WMT NVDA META
```

Scores each ticker with both FinSight and GPT-4o (given identical retrieved context and
signals), writes `eval/benchmark_results.csv`, and prints summary stats (mean/max absolute
difference, correlation).

## Project layout

```
ingest/edgar.py     SEC EDGAR client (submissions, archives, XBRL company facts)
store/chroma.py     Chunking + embedding + ChromaDB persistence
score/scorer.py     Signal extraction + RAG + fine-tuned Llama scoring (+ heuristic fallback)
api/main.py         FastAPI app: /score/{ticker}, /query
dashboard/          React + TypeScript UI (Vite, Recharts)
eval/benchmark.py   FinSight vs. GPT-4o comparison across companies
config/settings.py  Shared pydantic-settings config (reads .env)
```

## Known limitations

- **Hybrid retrieval** uses BM25, which is fast but treats all tokenized text equally. Compound
  concepts or phrases that cross chunk boundaries may not rank well; semantic search often finds
  these more reliably. If hybrid retrieval misses results, try `use_hybrid: false` for pure
  semantic search, or raise `n_results` to retrieve more candidates before fusion.
- **Citation grounding** in `/query` is manual: the LLM is prompted to cite sources [1], [2] etc.,
  but there is no automated verification that the cited excerpts actually support the claim. The
  FinSight agent does not yet check claimed facts against retrieved chunks — see "Multi-Agent
  Critique Loop" improvements below.
- `risk_language` and `filing_sentiment` are lexicon-based heuristics over retrieved chunks,
  not a trained classifier — treat them as directional signals, not ground truth.
- `revenue_trend` only looks at the two most recent annual XBRL revenue figures; it doesn't
  smooth over one-off items or segment-level detail.
- The EDGAR client reads only the "recent" filings window (last ~1000 filings per company),
  which comfortably covers a scoring window but not a company's full multi-decade history.
