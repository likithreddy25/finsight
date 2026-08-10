"""Chunk filing text, embed it with sentence-transformers, and persist it in ChromaDB."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from config.settings import get_settings
from ingest.edgar import Filing

COLLECTION_NAME = "sec_filings"


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    distance: float
    citation_index: int = 0  # Position in retrieval results for inline citation [1], [2], etc.


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """Split text into overlapping, word-boundary-respecting chunks (character-based)."""
    words = re.split(r"(\s+)", text)
    chunks: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) > chunk_size and current.strip():
            chunks.append(current.strip())
            # start next chunk with the trailing overlap of the previous one
            current = current[-overlap:] if overlap else ""
        current += word
    if current.strip():
        chunks.append(current.strip())
    return chunks


class FilingStore:
    """Persistent, ticker-aware vector store for SEC filing chunks."""

    def __init__(self, persist_dir: str | None = None, embedding_model: str | None = None):
        settings = get_settings()
        self._client = chromadb.PersistentClient(path=persist_dir or settings.chroma_persist_dir)
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model or settings.embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_filing(self, filing: Filing, chunk_size: int = 1200, overlap: int = 200) -> int:
        """Chunk a filing's text and upsert it. Idempotent: re-ingesting a filing overwrites its chunks."""
        if not filing.text:
            raise ValueError("filing.text is empty — call fetch_document_text() before storing it")

        pieces = chunk_text(filing.text, chunk_size=chunk_size, overlap=overlap)
        if not pieces:
            return 0

        ids = [f"{filing.accession_number}_{i}" for i in range(len(pieces))]
        metadatas = [
            {
                "ticker": filing.ticker,
                "cik": filing.cik,
                "form_type": filing.form_type,
                "accession_number": filing.accession_number,
                "filing_date": filing.filing_date.isoformat(),
                "report_date": filing.report_date.isoformat() if filing.report_date else "",
                "document_url": filing.document_url,
                "items": filing.items,
                "chunk_index": i,
            }
            for i in range(len(pieces))
        ]
        self._collection.upsert(ids=ids, documents=pieces, metadatas=metadatas)
        return len(pieces)

    def add_filings(self, filings: list[Filing]) -> int:
        return sum(self.add_filing(f) for f in filings)

    def _get_all_chunks(self, ticker: str | None = None, form_types: list[str] | None = None) -> list[tuple[str, dict]]:
        """Fetch all chunks matching filters from ChromaDB for BM25 indexing."""
        where: dict = {}
        conditions = []
        if ticker:
            conditions.append({"ticker": ticker.upper()})
        if form_types:
            conditions.append({"form_type": {"$in": form_types}})
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        result = self._collection.get(where=where or None, include=["documents", "metadatas"])
        return list(zip(result["documents"], result["metadatas"]))

    def _bm25_search(
        self, query_text: str, chunks: list[tuple[str, dict]], n_results: int = 8
    ) -> list[tuple[int, float]]:
        """BM25 keyword search over chunks. Returns list of (chunk_index, score) sorted by score."""
        if not chunks:
            return []
        docs = [chunk[0].split() for chunk in chunks]
        bm25 = BM25Okapi(docs)
        query_tokens = query_text.lower().split()
        scores = bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n_results]
        return [(idx, score) for idx, score in ranked if score > 0]

    def _reciprocal_rank_fusion(
        self, semantic_results: dict[int, float], bm25_results: dict[int, float], k: int = 60
    ) -> list[int]:
        """Merge semantic and BM25 rankings using reciprocal rank fusion (RRF).
        RRF score = sum(1 / (k + rank)) for each ranking that includes the item.
        k=60 is a common default that balances equal contribution from both rankers."""
        rrf_scores: dict[int, float] = {}

        for rank, idx in enumerate(sorted(semantic_results.keys(), key=lambda x: semantic_results[x], reverse=True)):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)

        for rank, idx in enumerate(sorted(bm25_results.keys(), key=lambda x: bm25_results[x], reverse=True)):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)

        return sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    def query_hybrid(
        self,
        query_text: str,
        ticker: str | None = None,
        form_types: list[str] | None = None,
        n_results: int = 8,
    ) -> list[RetrievedChunk]:
        """Hybrid retrieval: combine semantic search (ChromaDB) with BM25 keyword search.

        Uses reciprocal rank fusion to merge rankings from both methods, giving equal
        weight to semantic relevance and keyword matching. This helps surface results
        that might be missed by either method alone."""
        all_chunks = self._get_all_chunks(ticker=ticker, form_types=form_types)
        if not all_chunks:
            return []

        bm25_results_list = self._bm25_search(query_text, all_chunks, n_results=n_results * 2)
        bm25_results = {idx: score for idx, score in bm25_results_list}

        where: dict = {}
        conditions = []
        if ticker:
            conditions.append({"ticker": ticker.upper()})
        if form_types:
            conditions.append({"form_type": {"$in": form_types}})
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        semantic_result = self._collection.query(
            query_texts=[query_text],
            n_results=n_results * 2,
            where=where or None,
        )

        semantic_results = {
            i: (1 - dist) for i, dist in enumerate(semantic_result["distances"][0])
        }

        merged_indices = self._reciprocal_rank_fusion(semantic_results, bm25_results)[:n_results]

        chunks: list[RetrievedChunk] = []
        for citation_idx, chunk_idx in enumerate(merged_indices):
            text, metadata = all_chunks[chunk_idx]
            distance = 1 - semantic_results.get(chunk_idx, 0)
            chunks.append(
                RetrievedChunk(
                    text=text,
                    metadata=metadata,
                    distance=distance,
                    citation_index=citation_idx + 1,
                )
            )
        return chunks

    def query(
        self,
        query_text: str,
        ticker: str | None = None,
        form_types: list[str] | None = None,
        n_results: int = 8,
    ) -> list[RetrievedChunk]:
        """Semantic search only (original behavior). Use query_hybrid() for keyword+semantic."""
        where: dict = {}
        conditions = []
        if ticker:
            conditions.append({"ticker": ticker.upper()})
        if form_types:
            conditions.append({"form_type": {"$in": form_types}})
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        result = self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where or None,
        )
        chunks: list[RetrievedChunk] = []
        for citation_idx, (text, metadata, distance) in enumerate(
            zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
        ):
            chunks.append(
                RetrievedChunk(
                    text=text, metadata=metadata, distance=distance, citation_index=citation_idx + 1
                )
            )
        return chunks

    def has_filing(self, accession_number: str) -> bool:
        existing = self._collection.get(ids=[f"{accession_number}_0"])
        return len(existing["ids"]) > 0

    def count(self) -> int:
        return self._collection.count()


if __name__ == "__main__":
    import argparse

    from ingest.edgar import EdgarClient

    parser = argparse.ArgumentParser(description="Ingest a ticker's recent 8-K/10-K filings into ChromaDB.")
    parser.add_argument("ticker")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    store = FilingStore()
    with EdgarClient() as client:
        filings = client.get_filings_with_text(args.ticker, limit=args.limit)
    added = store.add_filings(filings)
    print(f"Stored {added} chunks from {len(filings)} filings for {args.ticker.upper()}")
    print(f"Collection now has {store.count()} total chunks")
