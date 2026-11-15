# Copyright 2026 Tauric Research
# SPDX-License-Identifier: Apache-2.0

"""SEC EDGAR fallback for XBRL financials, filings, and ownership when ``TAURIC_API_KEY`` is unset.

Covers:
- ``GET /securities/{ticker}/financials`` (XBRL companyfacts from ``data.sec.gov``,
  powering ``statements.py``, ``data.equity.fundamentals``, ``data.equity.earnings``,
  ``data.equity.ratios``, and ``data.equity.comps``)
- ``GET /securities/{ticker}/filings`` (SEC submissions history from ``data.sec.gov``,
  powering ``data.equity.filings``)
- ``GET /securities/{ticker}/insider-transactions`` (SEC Form 4 XML filings from EDGAR,
  powering ``data.equity.ownership`` insiders view)
- ``GET /securities/{ticker}/institutional-holders`` (13F institutional holdings,
  powering ``data.equity.ownership`` holders view)
"""

from __future__ import annotations

import asyncio
import datetime as dt
import re
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import Any

import httpx

from trading_r1.contract.spec import ToolError
from trading_r1.state import credentials

_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_DEFAULT_SEC_UA = "Trading-R1/0.1 (financial-research-terminal; contact@tauric.ai)"

_SECURITY_EDGAR_RE = re.compile(
    r"^/securities/([^/]+)/(financials|filings|insider-transactions|institutional-holders)$"
)

_FORM4_CODE_MEANINGS = {
    "P": "Open-market purchase",
    "S": "Open-market sale",
    "A": "Grant or award",
    "M": "Option exercise or conversion",
    "F": "Tax withholding",
    "G": "Gift",
    "D": "Disposition to issuer",
    "C": "Conversion of derivative",
}

_tickers_lock = asyncio.Lock()
_ticker_to_cik: dict[str, str] | None = None

_facts_locks: dict[str, asyncio.Lock] = {}
_facts_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
_submissions_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
_CACHE_MAX = 32


def _sec_headers() -> dict[str, str]:
    ua = (credentials.overlay().get("TR1_SEC_USER_AGENT") or _DEFAULT_SEC_UA).strip()
    return {"User-Agent": ua, "Accept": "application/json"}


def _clean_ticker(ticker: str) -> str:
    bare = ticker.rpartition(":")[2] or ticker
    return bare.upper().strip()


async def _resolve_cik(ticker: str) -> str:
    """Resolve a ticker (e.g. ``AAPL``, ``BRK.B``, ``BRK-B``) to a 10-digit SEC CIK string."""
    global _ticker_to_cik
    t = _clean_ticker(ticker)
    if _ticker_to_cik is None:
        async with _tickers_lock:
            if _ticker_to_cik is None:
                try:
                    async with httpx.AsyncClient(headers=_sec_headers(), timeout=15.0) as client:
                        resp = await client.get(_SEC_TICKERS_URL)
                        resp.raise_for_status()
                        raw = resp.json()
                except httpx.HTTPError as err:
                    raise ToolError(
                        f"SEC EDGAR ticker directory unavailable ({type(err).__name__})",
                        retry="later",
                    ) from err
                mapping: dict[str, str] = {}
                for entry in raw.values() if isinstance(raw, dict) else ():
                    sym = (entry.get("ticker") or "").upper().strip()
                    cik_num = entry.get("cik_str")
                    if sym and cik_num is not None:
                        cik10 = f"{int(cik_num):010d}"
                        mapping[sym] = cik10
                        mapping[sym.replace("-", ".")] = cik10
                        mapping[sym.replace(".", "-")] = cik10
                _ticker_to_cik = mapping
    cik = _ticker_to_cik.get(t) or _ticker_to_cik.get(t.replace(".", "-"))
    if not cik:
        raise ToolError(f"'{ticker}' is not a US SEC registrant", code="not_found")
    return cik


async def _load_company_facts(ticker: str) -> dict[str, Any]:
    cik = await _resolve_cik(ticker)
    if cik in _facts_cache:
        _facts_cache.move_to_end(cik)
        return _facts_cache[cik]
    lock = _facts_locks.setdefault(cik, asyncio.Lock())
    async with lock:
        if cik in _facts_cache:
            _facts_cache.move_to_end(cik)
            return _facts_cache[cik]
        try:
            async with httpx.AsyncClient(headers=_sec_headers(), timeout=20.0) as client:
                resp = await client.get(_SEC_FACTS_URL.format(cik=cik))
                if resp.status_code == 404:
                    raise ToolError(f"SEC holds no XBRL facts for '{ticker}'", code="not_found")
                resp.raise_for_status()
                body = resp.json()
        except ToolError:
            raise
        except httpx.HTTPError as err:
            raise ToolError(
                f"SEC EDGAR XBRL facts unavailable ({type(err).__name__})", retry="later"
            ) from err
        _facts_cache[cik] = body
        while len(_facts_cache) > _CACHE_MAX:
            _facts_cache.popitem(last=False)
        return body


async def _load_submissions(ticker: str) -> tuple[str, dict[str, Any]]:
    cik = await _resolve_cik(ticker)
    if cik in _submissions_cache:
        _submissions_cache.move_to_end(cik)
        return cik, _submissions_cache[cik]
    try:
        async with httpx.AsyncClient(headers=_sec_headers(), timeout=15.0) as client:
            resp = await client.get(_SEC_SUBMISSIONS_URL.format(cik=cik))
            if resp.status_code == 404:
                raise ToolError(f"no SEC submissions for '{ticker}'", code="not_found")
            resp.raise_for_status()
            body = resp.json()
    except ToolError:
        raise
    except httpx.HTTPError as err:
        raise ToolError(
            f"SEC EDGAR submissions unavailable ({type(err).__name__})", retry="later"
        ) from err
    _submissions_cache[cik] = body
    while len(_submissions_cache) > _CACHE_MAX:
        _submissions_cache.popitem(last=False)
    return cik, body


def _matches_period_type(period_type: str, is_instant: bool, duration_days: int, form: str) -> bool:
    if not form.startswith(("10-K", "10-Q", "20-F", "40-F")):
        return False
    if is_instant:
        if period_type == "instant":
            return True
        if period_type == "annual":
            return form.startswith(("10-K", "20-F", "40-F"))
        return False
    if period_type == "annual":
        return form.startswith(("10-K", "20-F", "40-F")) and 330 <= duration_days <= 380
    if period_type == "quarterly":
        return 75 <= duration_days <= 105
    if period_type == "ytd":
        return 160 <= duration_days <= 290
    return False


def _extract_concept_facts(
    all_facts: dict[str, Any], concept: str, period_type: str, limit: int
) -> dict[str, Any] | None:
    taxonomy, _, tag = concept.partition(":")
    if not tag:
        taxonomy, tag = "us-gaap", concept
    tax_node = (all_facts.get("facts") or {}).get(taxonomy) or {}
    concept_node = tax_node.get(tag)
    if not isinstance(concept_node, dict):
        return None
    units_map = concept_node.get("units") or {}
    if not units_map:
        return None

    chosen_unit = next(
        (u for u in ("USD", "USD/shares", "shares") if u in units_map),
        next(iter(units_map)),
    )
    raw_items = units_map.get(chosen_unit) or []

    dedup: dict[tuple[str | None, str], dict[str, Any]] = {}
    for item in raw_items:
        val = item.get("val")
        end = item.get("end")
        form = (item.get("form") or "").upper()
        if val is None or not end:
            continue
        start = item.get("start")
        is_instant = not start or start == end
        duration = 0
        if not is_instant:
            try:
                duration = (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days
            except ValueError:
                continue
        if not _matches_period_type(period_type, is_instant, duration, form):
            continue
        filed = item.get("filed") or ""
        key = (start, end)
        prev = dedup.get(key)
        if prev is None or filed >= prev["filed"]:
            dedup[key] = {
                "value": float(val),
                "period_end": end,
                "period_start": start,
                "duration_days": duration,
                "form": form,
                "filed": filed,
            }

    if not dedup:
        return None
    sorted_vals = sorted(
        dedup.values(),
        key=lambda x: (x["period_end"], x["filed"]),
        reverse=True,
    )
    return {"unit": chosen_unit, "values": sorted_vals[:limit]}


async def _handle_financials(ticker: str, params: dict[str, Any]) -> dict[str, Any]:
    raw_concepts = params.get("concepts") or ""
    concepts = [c.strip() for c in raw_concepts.split(",") if c.strip()]
    period_type = params.get("period_type") or "annual"
    limit = int(params.get("limit") or 8)

    all_facts = await _load_company_facts(ticker)
    out_concepts: dict[str, Any] = {}
    for c in concepts:
        extracted = _extract_concept_facts(all_facts, c, period_type, limit)
        if extracted is not None:
            out_concepts[c] = extracted
    return {"concepts": out_concepts}


def _parse_8k_items(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str) and raw.strip():
        return [x.strip() for x in raw.split(",") if x.strip()]
    return []


async def _handle_filings(ticker: str, params: dict[str, Any]) -> dict[str, Any]:
    want_form = (params.get("form") or "").upper().strip() or None
    limit = int(params.get("limit") or 15)
    cik, sub = await _load_submissions(ticker)
    recent = (sub.get("filings") or {}).get("recent") or {}

    accessions = recent.get("accessionNumber") or []
    forms = recent.get("form") or []
    filing_dates = recent.get("filingDate") or []
    report_dates = recent.get("reportDate") or []
    docs = recent.get("primaryDocument") or []
    items_col = recent.get("items") or []

    cik_int = int(cik)
    rows: list[dict[str, Any]] = []
    for i, acc in enumerate(accessions):
        form = (forms[i] if i < len(forms) else "").upper().strip()
        if not form:
            continue
        if want_form and form != want_form:
            continue
        filed = filing_dates[i] if i < len(filing_dates) else None
        rdate = (report_dates[i] if i < len(report_dates) else None) or None
        doc = docs[i] if i < len(docs) else ""
        acc_clean = acc.replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{doc}"
            if doc
            else f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/"
        )
        items = _parse_8k_items(items_col[i] if i < len(items_col) else None)
        rows.append(
            {
                "form": form,
                "filed": filed,
                "report_date": rdate,
                "url": url,
                "accession": acc,
                "items": items,
            }
        )
        if len(rows) >= limit:
            break
    return {"data": rows}


def _xml_text(elem: ET.Element | None, path: str) -> str | None:
    if elem is None:
        return None
    node = elem.find(path)
    if node is not None and node.text:
        return node.text.strip()
    return None


def _parse_form4_xml(acc: str, xml_bytes: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = ET.fromstring(xml_bytes)
    owners: list[dict[str, Any]] = []
    for rpt in root.findall(".//reportingOwner"):
        name = _xml_text(rpt, "./reportingOwnerId/rptOwnerName")
        title = _xml_text(rpt, "./reportingOwnerRelationship/officerTitle")
        is_dir = _xml_text(rpt, "./reportingOwnerRelationship/isDirector") in ("1", "true", "True")
        is_ten = _xml_text(rpt, "./reportingOwnerRelationship/isTenPercentOwner") in (
            "1",
            "true",
            "True",
        )
        if name:
            owners.append(
                {
                    "name": name,
                    "officer_title": title,
                    "is_director": is_dir,
                    "is_ten_percent_owner": is_ten,
                }
            )
    rule_10b5_1 = _xml_text(root, ".//aff10b5One") in ("1", "true", "True")
    filing_meta = {"owners": owners, "rule_10b5_1": rule_10b5_1}

    txs: list[dict[str, Any]] = []
    for tag, is_deriv in (
        (".//nonDerivativeTransaction", False),
        (".//derivativeTransaction", True),
    ):
        for tx in root.findall(tag):
            tdate = _xml_text(tx, "./transactionDate/value")
            code = _xml_text(tx, "./transactionCoding/transactionCode")
            shares_s = _xml_text(tx, "./transactionAmounts/transactionShares/value")
            price_s = _xml_text(tx, "./transactionAmounts/transactionPricePerShare/value")
            ad_code = _xml_text(tx, "./transactionAmounts/transactionAcquiredDisposedCode/value")
            owned_s = _xml_text(
                tx, "./postTransactionAmounts/sharesOwnedFollowingTransaction/value"
            )
            if not tdate or not code:
                continue
            try:
                shares = float(shares_s) if shares_s else None
            except ValueError:
                shares = None
            try:
                price = float(price_s) if price_s else None
            except ValueError:
                price = None
            try:
                owned = float(owned_s) if owned_s else None
            except ValueError:
                owned = None
            txs.append(
                {
                    "accession": acc,
                    "kind": "transaction",
                    "transaction_date": tdate,
                    "code": code,
                    "code_meaning": _FORM4_CODE_MEANINGS.get(code, code),
                    "shares": int(round(shares)) if shares is not None else None,
                    "price": price,
                    "owned_following": int(round(owned)) if owned is not None else None,
                    "acquired_disposed": ad_code,
                    "derivative": is_deriv,
                }
            )
    return filing_meta, txs


async def _handle_insider_transactions(ticker: str, params: dict[str, Any]) -> dict[str, Any]:
    if params.get("after"):
        return {"coverage": {}, "filings": {}, "data": [], "next_cursor": None}
    since = params.get("start") or (dt.date.today() - dt.timedelta(days=183)).isoformat()
    until = params.get("end")
    cik, sub = await _load_submissions(ticker)
    recent = (sub.get("filings") or {}).get("recent") or {}

    accessions = recent.get("accessionNumber") or []
    forms = recent.get("form") or []
    filing_dates = recent.get("filingDate") or []
    docs = recent.get("primaryDocument") or []

    cik_int = int(cik)
    candidates: list[tuple[str, str]] = []
    earliest_date: str | None = None
    for i, acc in enumerate(accessions):
        form = (forms[i] if i < len(forms) else "").upper().strip()
        if form not in ("4", "4/A"):
            continue
        fdate = filing_dates[i] if i < len(filing_dates) else ""
        earliest_date = fdate if earliest_date is None else min(earliest_date, fdate)
        if fdate < since:
            continue
        if until and fdate > until:
            continue
        doc = docs[i] if i < len(docs) else ""
        # Strip XSLT stylesheet prefix so we fetch the raw XML document
        raw_doc = doc.rsplit("/", 1)[-1] if "/" in doc else doc
        if not raw_doc.endswith(".xml"):
            continue
        acc_clean = acc.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{raw_doc}"
        candidates.append((acc, url))
        if len(candidates) >= 25:
            break

    sem = asyncio.Semaphore(5)
    filings_map: dict[str, Any] = {}
    all_txs: list[dict[str, Any]] = []

    async with httpx.AsyncClient(
        headers={"User-Agent": _sec_headers()["User-Agent"], "Accept": "application/xml,text/xml"},
        timeout=15.0,
    ) as client:

        async def fetch_one(acc: str, url: str) -> None:
            async with sem:
                try:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        return
                    f_meta, txs = _parse_form4_xml(acc, resp.content)
                    filings_map[acc] = f_meta
                    all_txs.extend(txs)
                except Exception:  # noqa: BLE001
                    return

        await asyncio.gather(*(fetch_one(a, u) for a, u in candidates))

    filtered = [
        t
        for t in all_txs
        if t["transaction_date"] >= since and (not until or t["transaction_date"] <= until)
    ]
    filtered.sort(key=lambda t: t["transaction_date"], reverse=True)
    return {
        "coverage": {"from": earliest_date or since},
        "filings": filings_map,
        "data": filtered,
        "next_cursor": None,
    }


async def _handle_institutional_holders(ticker: str, params: dict[str, Any]) -> dict[str, Any]:
    inst_own: list[dict[str, Any]] = []
    try:
        from trading_r1.tools.data.fallback.yfinance_data import (
            _fetch_quote_summary,
            _fmt_epoch_date,
            _raw_val,
            _yahoo_ticker,
        )

        qs = await _fetch_quote_summary(_yahoo_ticker(ticker), "institutionOwnership")
        inst_own = (qs.get("institutionOwnership") or {}).get("ownershipList") or []
    except ImportError:
        _raw_val = lambda x: x.get("raw") if isinstance(x, dict) else x  # noqa: E731
        _fmt_epoch_date = lambda x: str(x)[:10] if x else None  # noqa: E731

    filings: dict[str, Any] = {}
    data: list[dict[str, Any]] = []
    report_period: str | None = None

    for idx, item in enumerate(inst_own):
        org = item.get("organization")
        shares = _raw_val(item.get("position"))
        val = _raw_val(item.get("value"))
        rdate = _fmt_epoch_date(item.get("reportDate"))
        if not org or not shares:
            continue
        report_period = report_period or rdate
        acc = f"13f-{idx}"
        filings[acc] = {
            "view_state": "current",
            "manager": {"cik": f"mgr-{idx}", "name": org},
            "security_totals": {
                "shares": int(shares),
                "value": float(val) if val is not None else 0.0,
                "position_lines": 1,
            },
            "filed": rdate,
            "report_date": rdate,
        }
        data.append({"accession": acc})

    period = params.get("period") or report_period or dt.date.today().isoformat()
    return {
        "coverage": {
            "period": period,
            "period_state": "complete" if data else "not_covered",
            "periods": [{"period": period}] if data else [],
        },
        "filings": filings,
        "data": data,
    }


async def handle(method: str, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Handle SEC EDGAR routes (financials, filings, ownership) when ``TAURIC_API_KEY`` is unset."""
    if method != "GET":
        return None
    if m := _SECURITY_EDGAR_RE.match(path):
        ticker, action = m.group(1), m.group(2)
        if action == "financials":
            return await _handle_financials(ticker, payload)
        if action == "filings":
            return await _handle_filings(ticker, payload)
        if action == "insider-transactions":
            return await _handle_insider_transactions(ticker, payload)
        if action == "institutional-holders":
            return await _handle_institutional_holders(ticker, payload)
    return None
