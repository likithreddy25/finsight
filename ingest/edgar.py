"""Client for fetching real 8-K and 10-K filings from the SEC EDGAR REST APIs.

SEC's fair-access policy requires a descriptive User-Agent and asks clients to
stay at or below ~10 requests/second: https://www.sec.gov/os/webmaster-faq#developers
"""
from __future__ import annotations

import re
import time
import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import httpx
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import get_settings

# SEC filings are served as .htm but frequently wrap inline XBRL with dense custom
# namespaces; we intentionally parse them as HTML, so this warning is expected noise.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

FormType = Literal["8-K", "10-K"]

_MIN_REQUEST_INTERVAL = 0.11  # stay comfortably under SEC's ~10 req/s guidance


@dataclass
class Filing:
    ticker: str
    cik: str
    accession_number: str
    form_type: str
    filing_date: date
    report_date: date | None
    primary_document: str
    document_url: str
    items: str = ""  # comma-separated 8-K item codes, e.g. "5.02,9.01"; empty for 10-K
    text: str = field(default="", repr=False)


class EdgarClient:
    """Thin, rate-limited client over the SEC EDGAR submissions + archives APIs."""

    def __init__(self, user_agent: str | None = None):
        settings = get_settings()
        self._headers = {"User-Agent": user_agent or settings.sec_user_agent}
        self._client = httpx.Client(headers=self._headers, timeout=30.0)
        self._last_request_ts = 0.0
        self._ticker_to_cik: dict[str, str] | None = None

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "EdgarClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    )
    def _get(self, url: str, **kwargs) -> httpx.Response:
        self._throttle()
        resp = self._client.get(url, **kwargs)
        self._last_request_ts = time.monotonic()
        if resp.status_code == 429:
            raise httpx.HTTPStatusError("rate limited by SEC", request=resp.request, response=resp)
        resp.raise_for_status()
        return resp

    def _load_ticker_map(self) -> dict[str, str]:
        if self._ticker_to_cik is None:
            data = self._get(SEC_TICKER_MAP_URL).json()
            self._ticker_to_cik = {
                entry["ticker"].upper(): str(entry["cik_str"]).zfill(10) for entry in data.values()
            }
        return self._ticker_to_cik

    def resolve_cik(self, ticker: str) -> str:
        """Look up the zero-padded 10-digit CIK for a ticker symbol."""
        cik = self._load_ticker_map().get(ticker.upper())
        if cik is None:
            raise ValueError(f"Unknown ticker: {ticker!r}")
        return cik

    def get_filings(
        self,
        ticker: str,
        form_types: tuple[FormType, ...] = ("10-K", "8-K"),
        limit: int = 10,
    ) -> list[Filing]:
        """Return recent filing metadata for a ticker, newest first, without document text.

        Only searches the "recent" window returned by the submissions API (last ~1000
        filings), which comfortably covers 8-K/10-K history for a typical scoring window.
        """
        cik = self.resolve_cik(ticker)
        data = self._get(SEC_SUBMISSIONS_URL.format(cik=cik)).json()
        recent = data["filings"]["recent"]

        filings: list[Filing] = []
        for i in range(len(recent["form"])):
            form = recent["form"][i]
            if form not in form_types:
                continue
            accession_no_dashes = recent["accessionNumber"][i].replace("-", "")
            primary_doc = recent["primaryDocument"][i]
            doc_url = f"{SEC_ARCHIVES_BASE}/{int(cik)}/{accession_no_dashes}/{primary_doc}"
            report_date_raw = recent["reportDate"][i]
            filings.append(
                Filing(
                    ticker=ticker.upper(),
                    cik=cik,
                    accession_number=recent["accessionNumber"][i],
                    form_type=form,
                    filing_date=date.fromisoformat(recent["filingDate"][i]),
                    report_date=date.fromisoformat(report_date_raw) if report_date_raw else None,
                    primary_document=primary_doc,
                    document_url=doc_url,
                    items=recent.get("items", [""] * len(recent["form"]))[i],
                )
            )
            if len(filings) >= limit:
                break
        return filings

    def get_company_facts(self, ticker: str) -> dict:
        """Return the raw XBRL "company facts" payload (all reported us-gaap tags) for a ticker."""
        cik = self.resolve_cik(ticker)
        return self._get(SEC_COMPANYFACTS_URL.format(cik=cik)).json()

    def fetch_document_text(self, filing: Filing) -> str:
        """Download a filing's primary document and populate/return its cleaned plain text."""
        resp = self._get(filing.document_url)
        soup = BeautifulSoup(resp.text, "lxml")
        # ix:header is the hidden inline-XBRL fact block; it is machine data, not filing prose,
        # and ranks first in retrieval if left in.
        for tag in soup(["script", "style", "ix:header"]):
            tag.decompose()
        for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{2,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        filing.text = text.strip()
        return filing.text

    def get_filings_with_text(
        self,
        ticker: str,
        form_types: tuple[FormType, ...] = ("10-K", "8-K"),
        limit: int = 10,
    ) -> list[Filing]:
        """Convenience wrapper: fetch filing metadata and download each document's text."""
        filings = self.get_filings(ticker, form_types=form_types, limit=limit)
        for filing in filings:
            self.fetch_document_text(filing)
        return filings


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch SEC EDGAR filings for a ticker.")
    parser.add_argument("ticker")
    parser.add_argument("--forms", nargs="+", default=["10-K", "8-K"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--with-text", action="store_true")
    args = parser.parse_args()

    with EdgarClient() as edgar_client:
        if args.with_text:
            results = edgar_client.get_filings_with_text(args.ticker, tuple(args.forms), args.limit)
        else:
            results = edgar_client.get_filings(args.ticker, tuple(args.forms), args.limit)

    for f in results:
        print(f"{f.form_type:6s} {f.filing_date} {f.accession_number} {f.document_url}")
        if args.with_text:
            print(f"  chars={len(f.text)}")
