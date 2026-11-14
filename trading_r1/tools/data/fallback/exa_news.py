# Copyright 2026 Tauric Research
# SPDX-License-Identifier: Apache-2.0

"""Exa-backed fallback for Tauric news and digest endpoints when ``TAURIC_API_KEY`` is unset.

Covers:
- ``GET /news`` and ``GET /news/search`` (used by ``data.market.news``)
- ``GET /news/{id}`` (used by ``articles.resolve``)
- ``GET /securities/{ticker}/digest`` and ``GET /securities/{ticker}/thesis``
  (used by ``data.equity.digest``)
- ``GET /digests/market`` (used by ``data.market.digest``)
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from collections import OrderedDict
from typing import Any
from urllib.parse import urlparse

import httpx

from trading_r1.contract.spec import ToolError
from trading_r1.tools.web.search_engines import (
    cache_exa_text,
    exa_base_url,
    resolve_exa_api_key,
)

_ARTICLE_CACHE_MAX = 1024
_ARTICLES_BY_ID: OrderedDict[int, dict[str, Any]] = OrderedDict()

_BULLISH_WORDS = frozenset(
    {
        "beat",
        "beats",
        "surge",
        "surges",
        "rally",
        "rallies",
        "upgrade",
        "upgraded",
        "record",
        "growth",
        "outperform",
        "bullish",
        "gain",
        "gains",
        "soar",
        "soars",
    }
)
_BEARISH_WORDS = frozenset(
    {
        "miss",
        "misses",
        "plunge",
        "plunges",
        "drop",
        "drops",
        "downgrade",
        "downgraded",
        "lawsuit",
        "probe",
        "cut",
        "cuts",
        "bearish",
        "fall",
        "falls",
        "slump",
        "warning",
    }
)

_SECURITY_DIGEST_RE = re.compile(r"^/securities/([^/]+)/(digest|thesis)$")
_NEWS_ID_RE = re.compile(r"^/news/(\d+)$")


def _article_id(url: str) -> int:
    digest = hashlib.sha256(url.encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "big") & 0x7FFFFFFF) or 1


def _publisher(url: str, author: str | None = None) -> str:
    host = (urlparse(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host or (author or "web").strip()


def _infer_sentiment(text: str) -> str:
    words = set(re.findall(r"[a-z]+", text.lower()))
    bull = len(words & _BULLISH_WORDS)
    bear = len(words & _BEARISH_WORDS)
    if bull > 0 and bear > 0:
        return "mixed"
    if bull > bear:
        return "bullish"
    if bear > bull:
        return "bearish"
    return "neutral"


def _store_article(raw_article: dict[str, Any]) -> None:
    aid = raw_article["id"]
    _ARTICLES_BY_ID[aid] = raw_article
    _ARTICLES_BY_ID.move_to_end(aid)
    while len(_ARTICLES_BY_ID) > _ARTICLE_CACHE_MAX:
        _ARTICLES_BY_ID.popitem(last=False)


def _to_tauric_article(hit: dict[str, Any]) -> dict[str, Any] | None:
    url = (hit.get("url") or "").strip()
    title = (hit.get("title") or "").strip()
    if not url or not title:
        return None
    text = (hit.get("text") or "").strip()
    if text:
        cache_exa_text(url, text)
    highlights = [
        h.strip() for h in (hit.get("highlights") or []) if isinstance(h, str) and h.strip()
    ]
    summary = " - ".join(highlights) if highlights else text[:280]
    published = hit.get("publishedDate")
    time_kind = "published" if published else "discovered"
    if not published:
        published = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    sentiment = _infer_sentiment(f"{title} {summary}")
    polarity_article = {
        "id": _article_id(url),
        "title": title,
        "publisher": _publisher(url, hit.get("author")),
        "article_time": published,
        "article_time_kind": time_kind,
        "url": url,
        "annotations": {
            "summary": summary or title,
            "sentiment": sentiment,
            "impact": "medium",
            "event_types": ["news"],
        },
    }
    _store_article(polarity_article)
    return polarity_article


async def _exa_news_search(
    query: str,
    *,
    num_results: int = 21,
    since: str | None = None,
    until: str | None = None,
) -> list[dict[str, Any]]:
    api_key = resolve_exa_api_key()
    if not api_key:
        raise ToolError(
            "Neither TAURIC_API_KEY nor EXA_API_KEY is set. Set EXA_API_KEY "
            "(`trading-r1 secrets set EXA_API_KEY`) or TAURIC_API_KEY to use news tools."
        )
    body: dict[str, Any] = {
        "query": query,
        "category": "news",
        "type": "auto",
        "numResults": max(1, min(num_results, 25)),
        "contents": {
            "text": {"maxCharacters": 12000},
            "highlights": {"numSentences": 2, "highlightsPerUrl": 2},
        },
    }
    if since:
        body["startPublishedDate"] = since
    if until:
        body["endPublishedDate"] = until

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(f"{exa_base_url()}/search", headers=headers, json=body)
            if resp.status_code in (401, 403):
                raise ToolError("Exa rejected EXA_API_KEY (HTTP 401/403)")
            resp.raise_for_status()
            rows = resp.json().get("results") or []
    except ToolError:
        raise
    except httpx.HTTPError as err:
        raise ToolError(
            f"Exa news search unavailable ({type(err).__name__})", retry="later"
        ) from err

    articles: list[dict[str, Any]] = []
    for hit in rows:
        art = _to_tauric_article(hit)
        if art is not None:
            articles.append(art)
    return articles


def _ticker_from_instrument(instrument: str | None) -> str | None:
    if not instrument:
        return None
    return instrument.partition(":")[2] or instrument


async def _handle_news(path: str, params: dict[str, Any]) -> dict[str, Any]:
    ticker = _ticker_from_instrument(params.get("instrument"))
    raw_query = (params.get("q") or "").strip()
    if ticker and raw_query:
        query = f"{ticker} {raw_query}"
    elif ticker:
        query = f"{ticker} stock financial news"
    elif raw_query:
        query = raw_query
    else:
        query = "global financial markets stocks economy earnings news"

    limit = int(params.get("limit") or 21)
    articles = await _exa_news_search(
        query,
        num_results=limit,
        since=params.get("since"),
        until=params.get("until"),
    )
    wanted_sentiment = params.get("sentiment")
    if wanted_sentiment:
        articles = [
            a for a in articles if (a.get("annotations") or {}).get("sentiment") == wanted_sentiment
        ]
    if path == "/news/search":
        return {"data": [{"article": a} for a in articles]}
    return {"data": articles}


async def _handle_security_digest(ticker: str, view: str, params: dict[str, Any]) -> dict[str, Any]:
    day = params.get("date")
    since = f"{day}T00:00:00Z" if day else None
    until = f"{day}T23:59:59Z" if day else None
    query = (
        f"{ticker} stock news earnings guidance analyst"
        if view == "digest"
        else f"{ticker} investment thesis growth drivers risks catalysts"
    )
    articles = await _exa_news_search(query, num_results=10, since=since, until=until)
    if not articles:
        raise ToolError(f"no {view} for '{ticker}'", code="not_found")
    digest_date = day or (articles[0].get("article_time") or "")[:10] or dt.date.today().isoformat()
    polarity_map = {
        "bullish": "positive",
        "bearish": "negative",
        "mixed": "neutral",
        "neutral": "neutral",
    }
    if view == "digest":
        bullets = []
        for a in articles[:6]:
            ann = a.get("annotations") or {}
            bullets.append(
                {
                    "polarity": polarity_map.get(ann.get("sentiment") or "neutral", "neutral"),
                    "event_type": "news",
                    "text": f"{a['title']}: {ann.get('summary') or a['title']}",
                    "article_ids": [a["id"]],
                }
            )
        content: dict[str, Any] = {
            "headline": articles[0]["title"],
            "bullets": bullets,
            "watch_items": [a["title"] for a in articles[6:9]],
        }
    else:
        drivers = [
            {
                "text": (
                    f"{a['title']}: {(a.get('annotations') or {}).get('summary') or a['title']}"
                ),
                "article_ids": [a["id"]],
            }
            for a in articles[:5]
        ]
        risks = [
            {
                "text": (
                    f"{a['title']}: {(a.get('annotations') or {}).get('summary') or a['title']}"
                ),
                "article_ids": [a["id"]],
            }
            for a in articles
            if (a.get("annotations") or {}).get("sentiment") == "bearish"
        ][:3]
        content = {
            "narrative": (articles[0].get("annotations") or {}).get("summary")
            or articles[0]["title"],
            "drivers": drivers,
            "risks": risks,
            "catalysts": [],
            "open_questions": [],
        }
    return {
        "digest_date": digest_date,
        "evidence": {"cutoff": digest_date},
        "content": content,
    }


async def _handle_market_digest(params: dict[str, Any]) -> dict[str, Any]:
    day = params.get("date")
    since = f"{day}T00:00:00Z" if day else None
    until = f"{day}T23:59:59Z" if day else None
    articles = await _exa_news_search(
        "stock market economy federal reserve earnings headlines",
        num_results=10,
        since=since,
        until=until,
    )
    if not articles:
        raise ToolError("no market digest for that day", code="not_found")
    digest_date = day or (articles[0].get("article_time") or "")[:10] or dt.date.today().isoformat()
    polarity_map = {
        "bullish": "positive",
        "bearish": "negative",
        "mixed": "neutral",
        "neutral": "neutral",
    }
    top_stories = []
    for a in articles[:6]:
        ann = a.get("annotations") or {}
        top_stories.append(
            {
                "polarity": polarity_map.get(ann.get("sentiment") or "neutral", "neutral"),
                "event_type": "news",
                "text": f"{a['title']}: {ann.get('summary') or a['title']}",
                "article_ids": [a["id"]],
                "instruments": [],
            }
        )
    return {
        "digest_date": digest_date,
        "authorship": "Exa news",
        "content": {
            "headline": articles[0]["title"],
            "themes": [
                {
                    "name": "market_overview",
                    "summary": (articles[0].get("annotations") or {}).get("summary")
                    or articles[0]["title"],
                }
            ],
            "top_stories": top_stories,
            "watch_items": [a["title"] for a in articles[6:9]],
            "instruments": [],
        },
    }


async def handle(method: str, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Handle news and digest routes via Exa when ``TAURIC_API_KEY`` is not set."""
    if method != "GET":
        return None
    if path in ("/news", "/news/search"):
        return await _handle_news(path, payload)
    if m := _NEWS_ID_RE.match(path):
        aid = int(m.group(1))
        stored = _ARTICLES_BY_ID.get(aid)
        if stored is None:
            raise ToolError(f"article {aid} not found", code="not_found")
        return stored
    if m := _SECURITY_DIGEST_RE.match(path):
        ticker, view = m.group(1), m.group(2)
        return await _handle_security_digest(ticker, view, payload)
    if path == "/digests/market":
        return await _handle_market_digest(payload)
    return None
