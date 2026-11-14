# Copyright 2026 Tauric Research
# SPDX-License-Identifier: Apache-2.0

"""Web tools: search (discovery), visit (retrieval), compose_citations.

Built per run via factories that share a :class:`FetchCache`.
"""

from trading_r1.tools.web.cache import Citation, FetchCache
from trading_r1.tools.web.citations import build_citations_spec
from trading_r1.tools.web.registry import build_research_registry, citation_finalizer
from trading_r1.tools.web.search import build_search_spec
from trading_r1.tools.web.search_engines import (
    DdgsEngine,
    ExaEngine,
    SearchEngine,
    SearchResult,
    StubEngine,
    default_search_engine,
)
from trading_r1.tools.web.visit import build_visit_spec
from trading_r1.tools.web.visit_fetch import FetchedPage

__all__ = [
    "Citation",
    "DdgsEngine",
    "ExaEngine",
    "FetchCache",
    "FetchedPage",
    "SearchEngine",
    "SearchResult",
    "StubEngine",
    "build_citations_spec",
    "build_research_registry",
    "build_search_spec",
    "build_visit_spec",
    "citation_finalizer",
    "default_search_engine",
]
