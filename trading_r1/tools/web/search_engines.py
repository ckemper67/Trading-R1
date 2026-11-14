# Copyright 2026 Tauric Research
# SPDX-License-Identifier: Apache-2.0

"""Search engines: a pluggable interface with Exa (when configured) and ddgs default.

The deep-research `search` tool talks to an engine through this small protocol.
The core ships `ExaEngine` (neural/keyword search + full-text extraction via Exa
when `EXA_API_KEY` is set), `DdgsEngine` (the `ddgs` library, no infrastructure)
and `StubEngine` (canned results for offline tests).
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol

import httpx

from trading_r1.state import credentials

_EXA_DEFAULT_BASE_URL = "https://api.exa.ai"
_EXA_MAX_CHARS = 20000
_TEXT_CACHE_MAX = 512
_EXA_TEXT_CACHE: OrderedDict[str, str] = OrderedDict()


def resolve_exa_api_key() -> str | None:
    """Return the configured Exa API key from environment or credentials store, if any."""
    key = (credentials.overlay().get("EXA_API_KEY") or "").strip()
    return key or None


def exa_base_url() -> str:
    """Return the Exa API base URL."""
    env = credentials.overlay()
    return (env.get("TR1_EXA_BASE_URL") or _EXA_DEFAULT_BASE_URL).rstrip("/")


def cache_exa_text(url: str, text: str | None) -> None:
    """Store extracted page text from Exa so later `visit` calls hit memory."""
    cleaned = (text or "").strip()
    if not url or not cleaned:
        return
    _EXA_TEXT_CACHE[url] = cleaned
    _EXA_TEXT_CACHE.move_to_end(url)
    while len(_EXA_TEXT_CACHE) > _TEXT_CACHE_MAX:
        _EXA_TEXT_CACHE.popitem(last=False)


def get_cached_exa_text(url: str) -> str | None:
    """Return cached Exa page text for ``url``, if available."""
    text = _EXA_TEXT_CACHE.get(url)
    if text is not None:
        _EXA_TEXT_CACHE.move_to_end(url)
    return text


async def exa_fetch_text(url: str, *, timeout: float = 15.0) -> str | None:
    """Fetch page text for ``url`` via Exa's ``/contents`` endpoint when configured."""
    if cached := get_cached_exa_text(url):
        return cached
    api_key = resolve_exa_api_key()
    if not api_key:
        return None
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body = {
        "urls": [url],
        "text": {"maxCharacters": _EXA_MAX_CHARS},
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{exa_base_url()}/contents", headers=headers, json=body)
            if resp.status_code != 200:
                return None
            results = resp.json().get("results") or []
    except Exception:  # noqa: BLE001
        return None
    for item in results:
        text = (item.get("text") or "").strip()
        if text:
            cache_exa_text(url, text)
            return text
    return None


@dataclass(frozen=True)
class SearchResult:
    """One search hit returned by an engine."""

    title: str
    url: str
    snippet: str
    text: str | None = None


class SearchEngine(Protocol):
    """Minimal async protocol implemented by search providers."""

    name: str

    async def search(self, query: str, max_results: int) -> list[SearchResult]: ...


class DdgsEngine:
    """DuckDuckGo via the `ddgs` library. Synchronous under the hood, so the
    blocking call runs in a worker thread."""

    name = "ddgs"

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        return await asyncio.to_thread(self._search_sync, query, max_results)

    @staticmethod
    def _search_sync(query: str, max_results: int) -> list[SearchResult]:
        from ddgs import DDGS

        rows = DDGS().text(query, max_results=max_results)
        out: list[SearchResult] = []
        for r in rows:
            url = r.get("href") or r.get("url") or ""
            if not url:
                continue
            out.append(
                SearchResult(
                    title=(r.get("title") or "").strip(),
                    url=url,
                    snippet=(r.get("body") or "").strip(),
                )
            )
        return out


class ExaEngine:
    """Exa search engine (`POST /search`) with inline content extraction and ddgs fallback."""

    name = "exa"

    def __init__(self, api_key: str | None = None, *, timeout: float = 20.0) -> None:
        self._api_key = api_key
        self._timeout = timeout

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        api_key = self._api_key or resolve_exa_api_key()
        if not api_key:
            return await DdgsEngine().search(query, max_results)
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "query": query,
            "numResults": max_results,
            "type": "auto",
            "contents": {
                "text": {"maxCharacters": _EXA_MAX_CHARS},
                "highlights": {"numSentences": 3, "highlightsPerUrl": 2},
            },
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{exa_base_url()}/search", headers=headers, json=payload)
            resp.raise_for_status()
            rows = resp.json().get("results") or []

        out: list[SearchResult] = []
        for r in rows:
            url = (r.get("url") or "").strip()
            if not url:
                continue
            title = (r.get("title") or url).strip()
            text = (r.get("text") or "").strip()
            highlights = [
                h.strip() for h in (r.get("highlights") or []) if isinstance(h, str) and h.strip()
            ]
            snippet = " ".join(highlights) if highlights else text[:400]
            if text:
                cache_exa_text(url, text)
            out.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    text=text or None,
                )
            )
        return out


def default_search_engine() -> SearchEngine:
    """Return ``ExaEngine`` when ``EXA_API_KEY`` is configured, else ``DdgsEngine``."""
    if resolve_exa_api_key():
        return ExaEngine()
    return DdgsEngine()


@dataclass
class StubEngine:
    """Canned results for offline tests. ``by_query`` maps a query to results;
    a query with no entry returns ``default``."""

    name: str = "stub"
    by_query: dict[str, list[SearchResult]] | None = None
    default: list[SearchResult] | None = None

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        results = (self.by_query or {}).get(query, self.default or [])
        return results[:max_results]
