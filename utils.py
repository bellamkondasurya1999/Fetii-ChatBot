# utils.py
from __future__ import annotations
import re
import pandas as pd
from typing import Optional, Dict, Any, Tuple

def month_bounds(now: pd.Timestamp, months_back: int = 1) -> Tuple[pd.Timestamp, pd.Timestamp]:
    now = pd.Timestamp(now)
    first_this = now.normalize().replace(day=1)
    start = (first_this - pd.offsets.MonthBegin(months_back)).normalize()
    end = first_this
    return start, end

def _colon_to_dollar(sql: str) -> str:
    """
    Convert :named parameters (used in our code) to $named
    which is accepted by DuckDB on all versions.
    Example:  'ts >= :start AND ts < :end' -> 'ts >= $start AND ts < $end'
    """
    return re.sub(r":([A-Za-z_][A-Za-z0-9_]*)", r"$\1", sql)

def run_sql(con, sql: str, params: Optional[Dict[str, Any]] = None):
    """
    Execute SQL against a DuckDB connection with dict parameters.
    Accepts :named in the SQL and internally converts to $named.
    """
    params = params or {}
    sql2 = _colon_to_dollar(sql)
    return con.execute(sql2, params).df()

def format_sql_debug(sql: str, params: Optional[Dict[str, Any]] = None) -> str:
    """
    Show the SQL as executed (with $named) plus params.
    """
    sql2 = _colon_to_dollar(sql).strip()
    return sql2 + ("\n-- params: " + repr(params or {}))
