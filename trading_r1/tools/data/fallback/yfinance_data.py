# Copyright 2026 Tauric Research
# SPDX-License-Identifier: Apache-2.0

"""Yahoo Finance fallback for market and equity endpoints when ``TAURIC_API_KEY`` is unset.

Covers:
- ``GET /securities/{ticker}/history``, ``/series/{index}/history``,
  ``/futures/{root}/continuous/history``, ``/fx/{pair}/history``,
  ``/crypto/pairs/{pair}/history`` (used by ``pricing.py``)
- ``GET /securities/{ticker}/quote`` and ``POST /quotes``
- ``GET /instruments`` (used by ``data.market.symbol_search``)
- ``GET /markets/equity/movers`` (used by ``data.market.movers``)
- ``GET /series/vix-term-structure`` (used by ``data.market.vol_term_structure``)
- ``GET /securities/{ticker}/profile``, ``/analyst``, ``/calendar``, and
  ``/corporate-actions``
"""

from __future__ import annotations

import asyncio
import datetime as dt
import re
from typing import Any

import httpx

from trading_r1.contract.spec import ToolError
from trading_r1.tools.data.symbols import INDEX_SERIES

_YF_BASE = "https://query2.finance.yahoo.com"
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_HEADERS = {"User-Agent": _UA, "Accept": "application/json"}

_SERIES_TO_CARET: dict[str, str] = {v: k for k, v in INDEX_SERIES.items() if k != "^SPX"}

_HISTORY_RE = re.compile(
    r"^/(securities|series|futures|fx|crypto/pairs)/([^/]+)(?:/continuous)?/history$"
)
_SECURITY_SUB_RE = re.compile(
    r"^/securities/([^/]+)/(quote|profile|analyst|calendar|corporate-actions)$"
)

_VIX_TENORS = {
    "vix_9d": "^VIX9D",
    "spot": "^VIX",
    "vix_3m": "^VIX3M",
    "vix_6m": "^VIX6M",
    "vix_1y": "^VIX1Y",
}

_SCREENER_IDS = {
    "gainers": "day_gainers",
    "losers": "day_losers",
    "active": "most_actives",
}

# Cached Yahoo cookie + crumb session for quoteSummary calls
_crumb_lock = asyncio.Lock()
_crumb_state: tuple[str, httpx.Cookies] | None = None


def _yahoo_ticker(ticker: str) -> str:
    """Convert a Tauric canonical ticker (e.g. ``BRK.B`` or ``XNAS:AAPL``) to Yahoo format."""
    bare = ticker.rpartition(":")[2] or ticker
    return bare.replace(".", "-")


def _key_to_yahoo(key: str) -> tuple[str, str] | None:
    """Map a Tauric instrument key to ``(kind, yahoo_symbol)``."""
    kind, _, rest = key.partition(":")
    if kind == "security" and rest:
        return "security", _yahoo_ticker(rest)
    if kind == "series" and rest in _SERIES_TO_CARET:
        return "index", _SERIES_TO_CARET[rest]
    if kind == "future" and rest:
        root = rest.removesuffix("/continuous").rsplit(":", 1)[-1]
        return "future", f"{root}=F"
    if kind == "fx" and rest:
        return "fx", f"{rest}=X"
    if kind == "crypto" and rest:
        base, _, quote = rest.partition("-")
        y_quote = "USD" if quote in ("USDT", "USDC", "USD", "") else quote
        return "crypto", f"{base}-{y_quote}"
    return None


def _path_to_yahoo(route_group: str, ident: str) -> tuple[str, str, str] | None:
    """Map a history route ``(route_group, ident)`` to ``(kind, tauric_key, yahoo_symbol)``."""
    if route_group == "securities":
        return "security", f"security:{ident}", _yahoo_ticker(ident)
    if route_group == "series":
        if ident not in _SERIES_TO_CARET:
            return None
        return "index", f"series:{ident}", _SERIES_TO_CARET[ident]
    if route_group == "futures":
        return "future", f"future:{ident}/continuous", f"{ident}=F"
    if route_group == "fx":
        return "fx", f"fx:{ident}", f"{ident}=X"
    if route_group == "crypto/pairs":
        base, _, quote = ident.partition("-")
        y_quote = "USD" if quote in ("USDT", "USDC", "USD", "") else quote
        return "crypto", f"crypto:{ident}", f"{base}-{y_quote}"
    return None


def _raw_val(node: Any) -> Any:
    """Unwrap Yahoo quoteSummary ``{"raw": val, "fmt": ...}`` objects."""
    if isinstance(node, dict):
        return node.get("raw")
    return node


def _market_state(state: str | None) -> str:
    s = (state or "").upper()
    if s == "REGULAR":
        return "regular"
    if s in ("PRE", "PREPRE"):
        return "pre_market"
    if s in ("POST", "POSTPOST"):
        return "post_market"
    return "closed"


def _instrument_type(y_type: str | None) -> str:
    t = (y_type or "").upper()
    return {
        "EQUITY": "stock",
        "ETF": "etf",
        "MUTUALFUND": "fund",
        "INDEX": "index",
        "FUTURE": "future",
        "CURRENCY": "fx",
        "CRYPTOCURRENCY": "crypto",
    }.get(t, "stock")


async def _get_crumb_and_cookies() -> tuple[str | None, httpx.Cookies]:
    """Obtain and cache a Yahoo Finance session cookie + crumb for ``quoteSummary``."""
    global _crumb_state
    if _crumb_state is not None:
        return _crumb_state
    async with _crumb_lock:
        if _crumb_state is not None:
            return _crumb_state
        cookies = httpx.Cookies()
        try:
            async with httpx.AsyncClient(
                headers=_HEADERS, follow_redirects=True, timeout=10.0
            ) as client:
                await client.get("https://fc.yahoo.com")
                cookies.update(client.cookies)
                resp = await client.get(f"{_YF_BASE}/v1/test/getcrumb", cookies=cookies)
                if resp.status_code == 200 and resp.text and "<" not in resp.text:
                    _crumb_state = (resp.text.strip(), cookies)
                    return _crumb_state
        except Exception:  # noqa: BLE001
            pass
        return None, cookies


async def _fetch_chart(
    y_sym: str,
    *,
    interval: str = "1d",
    range_str: str | None = "1y",
    period1: int | None = None,
    period2: int | None = None,
    events: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"interval": interval, "includePrePost": "false"}
    if period1 is not None and period2 is not None:
        params["period1"] = period1
        params["period2"] = period2
    elif range_str is not None:
        params["range"] = range_str
    if events:
        params["events"] = events

    try:
        async with httpx.AsyncClient(headers=_HEADERS, timeout=15.0) as client:
            resp = await client.get(f"{_YF_BASE}/v8/finance/chart/{y_sym}", params=params)
            if resp.status_code == 404:
                raise ToolError(f"no data for '{y_sym}': check the symbol", code="not_found")
            resp.raise_for_status()
            body = resp.json()
    except ToolError:
        raise
    except httpx.HTTPError as err:
        raise ToolError(
            f"Yahoo Finance chart unavailable ({type(err).__name__})", retry="later"
        ) from err

    chart = body.get("chart") or {}
    if err := chart.get("error"):
        msg = err.get("description") or err.get("code") or f"no data for '{y_sym}'"
        raise ToolError(msg, code="not_found")
    results = chart.get("result") or []
    if not results:
        raise ToolError(f"no data for '{y_sym}'", code="not_found")
    return results[0]


async def _fetch_quote_summary(y_sym: str, modules: str) -> dict[str, Any]:
    crumb, cookies = await _get_crumb_and_cookies()
    params: dict[str, Any] = {"modules": modules}
    if crumb:
        params["crumb"] = crumb
    try:
        async with httpx.AsyncClient(headers=_HEADERS, cookies=cookies, timeout=15.0) as client:
            resp = await client.get(f"{_YF_BASE}/v10/finance/quoteSummary/{y_sym}", params=params)
            if resp.status_code in (401, 403):
                global _crumb_state
                _crumb_state = None
                crumb, cookies = await _get_crumb_and_cookies()
                if crumb:
                    params["crumb"] = crumb
                resp = await client.get(
                    f"{_YF_BASE}/v10/finance/quoteSummary/{y_sym}",
                    params=params,
                    cookies=cookies,
                )
            if resp.status_code == 404:
                raise ToolError(f"no data for '{y_sym}'", code="not_found")
            if resp.status_code >= 400:
                return {}
            body = resp.json()
    except ToolError:
        raise
    except Exception:  # noqa: BLE001
        return {}
    res = (body.get("quoteSummary") or {}).get("result") or []
    return res[0] if res else {}


def _period_to_days(period: str) -> int:
    mapping = {
        "1d": 5,
        "5d": 10,
        "1mo": 35,
        "3mo": 100,
        "6mo": 190,
        "1y": 370,
        "2y": 740,
        "5y": 1850,
        "10y": 3700,
        "max": 36500,
    }
    return mapping.get(period, 370)


async def _handle_history(
    kind: str, tauric_key: str, y_sym: str, params: dict[str, Any]
) -> dict[str, Any]:
    interval = params.get("interval") or "1d"
    intraday = interval in ("15m", "60m")
    end_str = params.get("end")
    end_dt = (
        dt.datetime.fromisoformat(f"{end_str}T23:59:59+00:00")
        if end_str
        else dt.datetime.now(dt.UTC)
    )
    if intraday:
        days = int(params.get("days") or 5)
        start_dt = end_dt - dt.timedelta(days=max(days + 4, 7))
        res = await _fetch_chart(
            y_sym,
            interval=interval,
            range_str=None,
            period1=int(start_dt.timestamp()),
            period2=int(end_dt.timestamp()),
        )
    else:
        period = params.get("period") or "1y"
        if end_str:
            days = _period_to_days(period)
            start_dt = end_dt - dt.timedelta(days=days)
            res = await _fetch_chart(
                y_sym,
                interval="1d",
                range_str=None,
                period1=int(start_dt.timestamp()),
                period2=int(end_dt.timestamp()),
            )
        else:
            yf_range = (
                period if period in ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max") else "5y"
            )
            res = await _fetch_chart(y_sym, interval="1d", range_str=yf_range)

    meta = res.get("meta") or {}
    stamps = res.get("timestamp") or []
    ind = res.get("indicators") or {}
    quotes = (ind.get("quote") or [{}])[0]
    adj_list = ((ind.get("adjclose") or [{}])[0]).get("adjclose") if not intraday else None

    opens = quotes.get("open") or []
    highs = quotes.get("high") or []
    lows = quotes.get("low") or []
    closes = quotes.get("close") or []
    volumes = quotes.get("volume") or []

    use_adj = bool(
        not intraday and kind == "security" and adj_list and len(adj_list) == len(closes)
    )
    bars: list[dict[str, Any]] = []
    for i, ts in enumerate(stamps):
        o = opens[i] if i < len(opens) else None
        h = highs[i] if i < len(highs) else None
        lo = lows[i] if i < len(lows) else None
        c = closes[i] if i < len(closes) else None
        v = volumes[i] if i < len(volumes) else None
        if c is None or o is None:
            continue
        if use_adj and adj_list[i] is not None and c > 0:
            factor = adj_list[i] / c
            o = o * factor
            h = h * factor if h is not None else None
            lo = lo * factor if lo is not None else None
            c = adj_list[i]
        moment = dt.datetime.fromtimestamp(ts, tz=dt.UTC)
        if intraday:
            bars.append(
                {
                    "start": moment.isoformat(timespec="seconds"),
                    "open": float(o),
                    "high": float(h) if h is not None else None,
                    "low": float(lo) if lo is not None else None,
                    "close": float(c),
                    "volume": int(v) if v is not None else None,
                }
            )
        else:
            bars.append(
                {
                    "date": moment.strftime("%Y-%m-%d"),
                    "open": float(o),
                    "high": float(h) if h is not None else None,
                    "low": float(lo) if lo is not None else None,
                    "close": float(c),
                    "volume": int(v) if v is not None else None,
                }
            )

    inst_type = _instrument_type(meta.get("instrumentType"))
    if y_sym in ("^IRX", "^FVX", "^TNX", "^TYX"):
        inst_type = "rate"
    return {
        "instrument": {
            "key": tauric_key,
            "currency": meta.get("currency") or "USD",
            "classification": {"instrument_type": inst_type},
        },
        "price_basis": "split_dividend_adjusted" if use_adj else "quoted",
        "bars": bars,
        "coverage": {"continue_end": None},
    }


async def _single_quote(tauric_key: str, y_sym: str) -> dict[str, Any]:
    res = await _fetch_chart(y_sym, interval="1d", range_str="5d")
    meta = res.get("meta") or {}
    ind = res.get("indicators") or {}
    q = (ind.get("quote") or [{}])[0]
    closes = [c for c in (q.get("close") or []) if c is not None]
    highs = [h for h in (q.get("high") or []) if h is not None]
    lows = [lo for lo in (q.get("low") or []) if lo is not None]
    vols = [v for v in (q.get("volume") or []) if v is not None]

    price = meta.get("regularMarketPrice")
    if price is None and closes:
        price = closes[-1]
    prev = meta.get("chartPreviousClose") or meta.get("previousClose")
    if prev is None and len(closes) >= 2:
        prev = closes[-2]
    change = (price - prev) if (price is not None and prev is not None) else None
    change_pct = (change / prev * 100.0) if (change is not None and prev) else None

    rm_time = meta.get("regularMarketTime")
    session_date = (
        dt.datetime.fromtimestamp(rm_time, tz=dt.UTC).strftime("%Y-%m-%d")
        if rm_time
        else dt.date.today().isoformat()
    )
    return {
        "instrument": {
            "key": tauric_key,
            "currency": meta.get("currency") or "USD",
            "name": meta.get("longName") or meta.get("shortName") or y_sym,
        },
        "price": float(price) if price is not None else None,
        "change": float(change) if change is not None else None,
        "change_percent": float(change_pct) if change_pct is not None else None,
        "previous_close": {"value": float(prev) if prev is not None else None},
        "day": {
            "high": float(meta.get("regularMarketDayHigh") or (highs[-1] if highs else 0.0))
            if (meta.get("regularMarketDayHigh") or highs)
            else None,
            "low": float(meta.get("regularMarketDayLow") or (lows[-1] if lows else 0.0))
            if (meta.get("regularMarketDayLow") or lows)
            else None,
            "volume": int(meta.get("regularMarketVolume") or (vols[-1] if vols else 0))
            if (meta.get("regularMarketVolume") or vols)
            else None,
        },
        "range_52w": {
            "high": meta.get("fiftyTwoWeekHigh"),
            "low": meta.get("fiftyTwoWeekLow"),
        },
        "market_state": _market_state(meta.get("marketState")),
        "source_time": session_date,
        "session_date": session_date,
    }


async def _handle_batch_quotes(payload: dict[str, Any]) -> dict[str, Any]:
    instruments: list[str] = payload.get("instruments") or []
    sem = asyncio.Semaphore(8)
    quotes: dict[str, dict[str, Any]] = {}
    errors: dict[str, dict[str, str]] = {}

    async def one(key: str) -> None:
        mapped = _key_to_yahoo(key)
        if mapped is None:
            errors[key] = {"message": f"unsupported instrument key '{key}'"}
            return
        _, y_sym = mapped
        async with sem:
            try:
                quotes[key] = await _single_quote(key, y_sym)
            except ToolError as err:
                errors[key] = {"message": str(err)}

    await asyncio.gather(*(one(k) for k in instruments))
    return {"quotes": quotes, "errors": errors}


async def _handle_symbol_search(params: dict[str, Any]) -> dict[str, Any]:
    query = params.get("q") or ""
    limit = int(params.get("limit") or 8)
    want_kind = params.get("kind")
    want_type = params.get("instrument_type")

    try:
        async with httpx.AsyncClient(headers=_HEADERS, timeout=15.0) as client:
            resp = await client.get(
                f"{_YF_BASE}/v1/finance/search",
                params={"q": query, "quotesCount": max(limit * 2, 16), "newsCount": 0},
            )
            resp.raise_for_status()
            rows = resp.json().get("quotes") or []
    except httpx.HTTPError as err:
        raise ToolError(
            f"Yahoo Finance symbol search unavailable ({type(err).__name__})", retry="later"
        ) from err

    out: list[dict[str, Any]] = []
    for r in rows:
        sym = (r.get("symbol") or "").strip().upper()
        if not sym:
            continue
        qtype = (r.get("quoteType") or "").upper()
        name = r.get("longname") or r.get("shortname") or sym
        if qtype in ("EQUITY", "ETF", "MUTUALFUND"):
            kind = "security"
            itype = _instrument_type(qtype)
            key = f"security:{sym.replace('-', '.')}"
        elif qtype == "INDEX":
            series_key = INDEX_SERIES.get(sym)
            if not series_key:
                continue
            kind = "series"
            itype = "index"
            key = f"series:{series_key}"
        elif qtype == "FUTURE" or sym.endswith("=F"):
            root = sym.removesuffix("=F")
            kind = "future"
            itype = "future"
            key = f"future:{root}/continuous"
        elif qtype == "CURRENCY" or sym.endswith("=X"):
            pair = sym.removesuffix("=X")
            if len(pair) == 3:
                pair = f"USD{pair}"
            kind = "fx"
            itype = "fx"
            key = f"fx:{pair}"
        elif qtype == "CRYPTOCURRENCY":
            base = sym.partition("-")[0]
            kind = "crypto"
            itype = "crypto"
            key = f"crypto:{base}-USDT"
        else:
            continue

        if want_kind and kind != want_kind:
            continue
        if want_type and itype != want_type:
            continue
        out.append(
            {
                "key": key,
                "name": name,
                "classification": {"instrument_type": itype},
            }
        )
        if len(out) >= limit:
            break
    return {"data": out}


async def _handle_movers(params: dict[str, Any]) -> dict[str, Any]:
    view = params.get("type") or "gainers"
    limit = int(params.get("limit") or 10)
    scr_id = _SCREENER_IDS.get(view, "day_gainers")

    try:
        async with httpx.AsyncClient(headers=_HEADERS, timeout=15.0) as client:
            resp = await client.get(
                f"{_YF_BASE}/v1/finance/screener/predefined/saved",
                params={"scrIds": scr_id, "count": max(limit, 20)},
            )
            resp.raise_for_status()
            body = resp.json()
    except httpx.HTTPError as err:
        raise ToolError(
            f"Yahoo Finance movers unavailable ({type(err).__name__})", retry="later"
        ) from err

    res = ((body.get("finance") or {}).get("result") or [{}])[0]
    quotes = res.get("quotes") or []
    today = dt.date.today().isoformat()
    rows: list[dict[str, Any]] = []
    for idx, q in enumerate(quotes[:limit], start=1):
        sym = (q.get("symbol") or "").replace("-", ".")
        if not sym:
            continue
        rows.append(
            {
                "rank": idx,
                "instrument": {
                    "key": f"security:{sym}",
                    "name": q.get("longName") or q.get("shortName") or sym,
                    "currency": q.get("currency") or "USD",
                },
                "price": _raw_val(q.get("regularMarketPrice")),
                "change": _raw_val(q.get("regularMarketChange")),
                "change_percent": _raw_val(q.get("regularMarketChangePercent")),
                "day": {"volume": _raw_val(q.get("regularMarketVolume"))},
                "avg_volume": _raw_val(q.get("averageDailyVolume3Month")),
                "market_cap": _raw_val(q.get("marketCap")),
                "market_state": _market_state(q.get("marketState")),
            }
        )
    total = res.get("total") or len(rows)
    return {
        "session_date": today,
        "population": {
            "members": total,
            "ranked": total,
            "definition": "US common shares and ADRs (Yahoo Finance screener)",
        },
        "qualifying": len(rows),
        "data": rows,
    }


async def _handle_vix_term_structure() -> dict[str, Any]:
    async def one(field: str, y_sym: str) -> tuple[str, dict[str, Any] | None, str | None]:
        try:
            res = await _fetch_chart(y_sym, interval="1d", range_str="1mo")
        except ToolError:
            return field, None, None
        q = ((res.get("indicators") or {}).get("quote") or [{}])[0]
        stamps = res.get("timestamp") or []
        closes = q.get("close") or []
        valid = [
            (
                dt.datetime.fromtimestamp(ts, tz=dt.UTC).strftime("%Y-%m-%d"),
                float(c),
            )
            for ts, c in zip(stamps, closes, strict=False)
            if c is not None
        ]
        if not valid:
            return field, None, None
        cur_date, cur_val = valid[-1]
        prev_val = valid[-2][1] if len(valid) >= 2 else None
        chg = (cur_val - prev_val) if prev_val is not None else 0.0
        five = None
        if len(valid) >= 6:
            d5, v5 = valid[-6]
            five = {"absolute": cur_val - v5, "from_period": {"label": d5}}
        return (
            field,
            {"price": cur_val, "change": chg, "change_5d": five},
            cur_date,
        )

    fetched = await asyncio.gather(*(one(f, s) for f, s in _VIX_TENORS.items()))
    out: dict[str, Any] = {}
    session_date = None
    for field, node, s_date in fetched:
        if node is not None:
            out[field] = node
            session_date = session_date or s_date

    spot = (out.get("spot") or {}).get("price")
    v3m = (out.get("vix_3m") or {}).get("price")
    slope = (v3m - spot) if (spot is not None and v3m is not None) else None
    structure = None
    if slope is not None:
        structure = "contango" if slope > 0 else "backwardation" if slope < 0 else "flat"
    out["session_date"] = session_date or dt.date.today().isoformat()
    out["slope"] = slope
    out["structure"] = structure
    return out


async def _handle_profile(ticker: str) -> dict[str, Any]:
    y_sym = _yahoo_ticker(ticker)
    mods = (
        "assetProfile,summaryProfile,summaryDetail,"
        "defaultKeyStatistics,financialData,price,quoteType"
    )
    qs, quote = await asyncio.gather(
        _fetch_quote_summary(y_sym, mods),
        _single_quote(f"security:{ticker}", y_sym),
    )
    ap = qs.get("assetProfile") or qs.get("summaryProfile") or {}
    sd = qs.get("summaryDetail") or {}
    ks = qs.get("defaultKeyStatistics") or {}
    fd = qs.get("financialData") or {}
    pr = qs.get("price") or {}
    qt = qs.get("quoteType") or {}

    de_raw = _raw_val(fd.get("debtToEquity"))
    debt_to_equity = (de_raw / 100.0) if de_raw is not None else None

    inst_name = (
        pr.get("longName")
        or pr.get("shortName")
        or qt.get("longName")
        or (quote.get("instrument") or {}).get("name")
        or ticker
    )
    currency = (
        pr.get("currency")
        or sd.get("currency")
        or (quote.get("instrument") or {}).get("currency")
        or "USD"
    )
    itype = _instrument_type(qt.get("quoteType"))
    return {
        "instrument": {
            "key": f"security:{ticker}",
            "name": inst_name,
            "currency": currency,
            "venue": pr.get("exchangeName") or pr.get("exchange"),
            "classification": {"instrument_type": itype},
        },
        "financial_currency": fd.get("financialCurrency") or currency,
        "market_cap_price_date": quote.get("session_date"),
        "sector": ap.get("sector"),
        "industry_group": ap.get("industry"),
        "industry": ap.get("industry"),
        "sub_industry": ap.get("industry"),
        "country": ap.get("country"),
        "business_summary": ap.get("longBusinessSummary"),
        "website": ap.get("website"),
        "employees": _raw_val(ap.get("fullTimeEmployees")),
        "market_cap": _raw_val(pr.get("marketCap")) or _raw_val(sd.get("marketCap")),
        "enterprise_value": _raw_val(ks.get("enterpriseValue")),
        "shares_outstanding": _raw_val(ks.get("sharesOutstanding")),
        "float_shares": _raw_val(ks.get("floatShares")),
        "beta": _raw_val(ks.get("beta")) or _raw_val(sd.get("beta")),
        "pe_ratio": _raw_val(sd.get("trailingPE")),
        "forward_pe": _raw_val(sd.get("forwardPE")) or _raw_val(ks.get("forwardPE")),
        "price_to_sales": _raw_val(sd.get("priceToSalesTrailing12Months")),
        "price_to_book": _raw_val(ks.get("priceToBook")),
        "peg_ratio": _raw_val(ks.get("pegRatio")),
        "dividend_yield": _raw_val(sd.get("dividendYield")),
        "payout_ratio": _raw_val(sd.get("payoutRatio")),
        "current_ratio": _raw_val(fd.get("currentRatio")),
        "quick_ratio": _raw_val(fd.get("quickRatio")),
        "debt_to_equity": debt_to_equity,
        "total_debt": _raw_val(fd.get("totalDebt")),
        "total_cash": _raw_val(fd.get("totalCash")),
        "return_on_assets": _raw_val(fd.get("returnOnAssets")),
        "return_on_equity": _raw_val(fd.get("returnOnEquity")),
        "operating_margin": _raw_val(fd.get("operatingMargins")),
        "profit_margin": _raw_val(fd.get("profitMargins")),
        "revenue_growth": _raw_val(fd.get("revenueGrowth")),
        "earnings_growth": _raw_val(fd.get("earningsGrowth")),
        "held_fraction_insiders": _raw_val(ks.get("heldPercentInsiders")),
        "held_fraction_institutions": _raw_val(ks.get("heldPercentInstitutions")),
        "shares_short": _raw_val(ks.get("sharesShort")),
        "days_to_cover": _raw_val(ks.get("shortRatio")),
        "short_fraction_of_float": _raw_val(ks.get("shortPercentOfFloat")),
    }


async def _handle_analyst(ticker: str) -> dict[str, Any]:
    y_sym = _yahoo_ticker(ticker)
    qs = await _fetch_quote_summary(
        y_sym, "financialData,defaultKeyStatistics,recommendationTrend,price"
    )
    fd = qs.get("financialData") or {}
    ks = qs.get("defaultKeyStatistics") or {}
    pr = qs.get("price") or {}
    trends = ((qs.get("recommendationTrend") or {}).get("trend") or [{}])[0]

    sb = trends.get("strongBuy")
    b = trends.get("buy")
    h = trends.get("hold")
    s = trends.get("sell")
    ss = trends.get("strongSell")
    counts = [x for x in (sb, b, h, s, ss) if isinstance(x, int)]
    total = sum(counts) if counts else None

    out = {
        "instrument": {
            "key": f"security:{ticker}",
            "currency": pr.get("currency") or fd.get("financialCurrency") or "USD",
        },
        "target_mean_price": _raw_val(fd.get("targetMeanPrice")),
        "target_median_price": _raw_val(fd.get("targetMedianPrice")),
        "target_high_price": _raw_val(fd.get("targetHighPrice")),
        "target_low_price": _raw_val(fd.get("targetLowPrice")),
        "number_of_analysts": _raw_val(fd.get("numberOfAnalystOpinions")),
        "recommendation": fd.get("recommendationKey"),
        "strong_buy": sb,
        "buy": b,
        "hold": h,
        "sell": s,
        "strong_sell": ss,
        "ratings_total": total,
        "trailing_eps": _raw_val(ks.get("trailingEps")),
        "forward_eps": _raw_val(ks.get("forwardEps")),
    }
    if all(
        out.get(k) is None
        for k in ("target_mean_price", "number_of_analysts", "trailing_eps", "forward_eps")
    ):
        raise ToolError(f"no analyst coverage for '{ticker}'", code="not_published")
    return out


def _fmt_epoch_date(val: Any) -> str | None:
    raw = _raw_val(val)
    if isinstance(raw, (int, float)):
        return dt.datetime.fromtimestamp(raw, tz=dt.UTC).strftime("%Y-%m-%d")
    if isinstance(raw, str) and len(raw) >= 10:
        return raw[:10]
    return None


async def _handle_calendar(ticker: str) -> dict[str, Any]:
    y_sym = _yahoo_ticker(ticker)
    qs = await _fetch_quote_summary(y_sym, "calendarEvents,summaryDetail")
    ce = qs.get("calendarEvents") or {}
    sd = qs.get("summaryDetail") or {}
    earn = ce.get("earnings") or {}
    edates = earn.get("earningsDate") or []
    edate = _fmt_epoch_date(edates[0]) if edates else None

    ex_div = _fmt_epoch_date(ce.get("exDividendDate") or sd.get("exDividendDate"))
    pay_div = _fmt_epoch_date(ce.get("dividendDate"))
    div_rate = _raw_val(sd.get("lastDividendValue"))
    dividends = []
    if ex_div:
        dividends.append({"ex_date": ex_div, "payment_date": pay_div, "amount": div_rate})
    return {
        "earnings": {
            "earnings_date": edate,
            "eps_estimate": _raw_val(earn.get("earningsAverage")),
            "revenue_estimate": _raw_val(earn.get("revenueAverage")),
        },
        "dividends": dividends,
    }


async def _handle_corporate_actions(ticker: str) -> dict[str, Any]:
    y_sym = _yahoo_ticker(ticker)
    res = await _fetch_chart(y_sym, interval="1mo", range_str="max", events="div,split,capitalGain")
    meta = res.get("meta") or {}
    currency = meta.get("currency") or "USD"
    events = res.get("events") or {}
    rows: list[dict[str, Any]] = []

    for item in (events.get("dividends") or {}).values():
        ts = item.get("date")
        amt = item.get("amount")
        if ts is not None and amt:
            rows.append(
                {
                    "ex_date": dt.datetime.fromtimestamp(ts, tz=dt.UTC).strftime("%Y-%m-%d"),
                    "action_type": "dividend",
                    "value": float(amt),
                    "currency": currency,
                }
            )
    for item in (events.get("splits") or {}).values():
        ts = item.get("date")
        num = item.get("numerator")
        den = item.get("denominator")
        if ts is not None and num and den:
            rows.append(
                {
                    "ex_date": dt.datetime.fromtimestamp(ts, tz=dt.UTC).strftime("%Y-%m-%d"),
                    "action_type": "split",
                    "value": float(num) / float(den),
                    "currency": currency,
                }
            )
    for item in (events.get("capitalGains") or {}).values():
        ts = item.get("date")
        amt = item.get("amount")
        if ts is not None and amt:
            rows.append(
                {
                    "ex_date": dt.datetime.fromtimestamp(ts, tz=dt.UTC).strftime("%Y-%m-%d"),
                    "action_type": "capital_gain",
                    "value": float(amt),
                    "currency": currency,
                }
            )
    return {
        "instrument": {"key": f"security:{ticker}", "currency": currency},
        "data": rows,
    }


async def handle(method: str, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Handle market and equity snapshot routes via Yahoo Finance."""
    if method == "POST" and path == "/quotes":
        return await _handle_batch_quotes(payload)
    if method != "GET":
        return None
    if m := _HISTORY_RE.match(path):
        mapped = _path_to_yahoo(m.group(1), m.group(2))
        if mapped is None:
            return None
        kind, tauric_key, y_sym = mapped
        return await _handle_history(kind, tauric_key, y_sym, payload)
    if m := _SECURITY_SUB_RE.match(path):
        ticker, action = m.group(1), m.group(2)
        if action == "quote":
            return await _single_quote(f"security:{ticker}", _yahoo_ticker(ticker))
        if action == "profile":
            return await _handle_profile(ticker)
        if action == "analyst":
            return await _handle_analyst(ticker)
        if action == "calendar":
            return await _handle_calendar(ticker)
        if action == "corporate-actions":
            return await _handle_corporate_actions(ticker)
    if path == "/instruments":
        return await _handle_symbol_search(payload)
    if path == "/markets/equity/movers":
        return await _handle_movers(payload)
    if path == "/series/vix-term-structure":
        return await _handle_vix_term_structure()
    return None
