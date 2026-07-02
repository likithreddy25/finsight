"""Chunk filing text, embed it with sentence-transformers, and persist it in ChromaDB."""
from __future__ import annotations

import re
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions

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

    def query(
        self,
        query_text: str,
        ticker: str | None = None,
        form_types: list[str] | None = None,
        n_results: int = 8,
    ) -> list[RetrievedChunk]:
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
        for text, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            chunks.append(RetrievedChunk(text=text, metadata=metadata, distance=distance))
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
