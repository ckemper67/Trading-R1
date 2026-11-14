# Copyright 2026 Tauric Research
# SPDX-License-Identifier: Apache-2.0

"""Keyless public-data fallback router for the Tauric data client.

When ``TAURIC_API_KEY`` is not set, ``Tauric.get`` and ``Tauric.post`` call
:func:`dispatch` before failing. Each domain fallback module lives alongside
this package and exposes ``async def handle(method, path, payload) -> dict | None``.
Modules are imported lazily so feature branches can be developed and merged
independently without conflicts.
"""

from __future__ import annotations

import importlib
from typing import Any

from trading_r1.contract.spec import ToolError

_MODULES: tuple[str, ...] = (
    "trading_r1.tools.data.fallback.exa_news",
    "trading_r1.tools.data.fallback.yfinance_data",
    "trading_r1.tools.data.fallback.sec_edgar",
    "trading_r1.tools.data.fallback.macro_rates",
)

_MISSING_KEY_MSG = (
    "TAURIC_API_KEY is not set. Market data needs it: register with Tauric, then "
    "`export TAURIC_API_KEY=...` or run "
    "`trading-r1 secrets set TAURIC_API_KEY`"
)


async def dispatch(method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Route a data request to the first installed fallback module that handles ``path``."""
    for mod_name in _MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        handler = getattr(mod, "handle", None)
        if handler is None:
            continue
        result = await handler(method, path, payload)
        if result is not None:
            return result
    raise ToolError(_MISSING_KEY_MSG)
