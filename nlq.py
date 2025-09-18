# nlq.py — Natural-Language → SQL for Fetii (VS Code friendly)
# ------------------------------------------------------------
# Usage:
#   from data_prep import load_trip_csv_to_duckdb
#   from nlq import parse_intent, execute_intent
#   con = load_trip_csv_to_duckdb("data/trip_data.csv")
#   msg, df, debug_sql = execute_intent(con, parse_intent("Top 10 drop-offs last week"))
#   print(msg); print(df); print(debug_sql)

from __future__ import annotations
import os
import re
import pandas as pd
from typing import Optional, Tuple, Dict, Any

from utils import month_bounds, run_sql, format_sql_debug


# ---------------- Brand (curated answers) ----------------
ABOUT_FETII = (
    "Fetii is the leading group rideshare platform—moving groups safely, efficiently, "
    "and socially. It reduces congestion and rideshare fragmentation by letting friends "
    "travel together with one tap. Fetii is backed by Y Combinator and Mark Cuban."
)

_BRAND_QA = [
    (re.compile(r"\b(what\s+is|tell\s+me\s+about)\s+(the\s+)?fetii\b"), ABOUT_FETII),
    (re.compile(r"\b(who\s+backs\s+fetii|invest(or|ment)s?)\b"), "Fetii is backed by Y Combinator and Mark Cuban."),
    (re.compile(r"\b(what\s+cities|where)\s+(does\s+)?fetii\s+(operate|available|cover|serve)\b"),
     "This demo uses Fetii’s Austin, TX dataset; other markets aren’t included here."),
    (re.compile(r"\b(pricing|price|cost|fare|fees?)\b"),
     "Pricing details are not included in this demo dataset."),
    (re.compile(r"\b(safe|safety)\b"),
     "Per the brief: Fetii moves people safely, efficiently, and socially. Detailed safety programs aren’t in the dataset."),
]

def _brand_answer(t: str) -> Optional[str]:
    low = t.lower()
    if re.search(r"\bfe+t+i+i?\b", low):
        for pat, ans in _BRAND_QA:
            if pat.search(low):
                return ans
        return ABOUT_FETII
    return None


# ---------------- Wikipedia (keyless fallback) ----------------
try:
    import wikipedia
    wikipedia.set_lang("en")
except Exception:  # library not installed or blocked
    wikipedia = None  # type: ignore

def _wiki_answer(query: str) -> Optional[str]:
    if wikipedia is None:
        return None
    try:
        return wikipedia.summary(query, sentences=3)
    except wikipedia.DisambiguationError as e:  # type: ignore[attr-defined]
        for opt in e.options[:5]:
            try:
                return wikipedia.summary(opt, sentences=3)
            except Exception:
                continue
        return None
    except wikipedia.PageError:  # type: ignore[attr-defined]
        try:
            for hit in wikipedia.search(query, results=3):
                try:
                    return wikipedia.summary(hit, sentences=3)
                except Exception:
                    continue
        except Exception:
            return None
    except Exception:
        return None


# ---------------- Tiny local model (guarded) ----------------
_LOCAL_PIPE = None
_BAD_TOKENS = {"fetish", "nsfw", "xxx"}

def _local_llm_answer(prompt: str) -> Optional[str]:
    """Use FLAN-T5 small locally if transformers is available."""
    global _LOCAL_PIPE
    try:
        if _LOCAL_PIPE is None:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
            tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
            mod = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
            _LOCAL_PIPE = pipeline("text2text-generation", model=mod, tokenizer=tok)
        out = _LOCAL_PIPE(prompt, max_new_tokens=220, do_sample=False)[0]["generated_text"]
        return (out or "").strip()
    except Exception:
        return None

def _guard(ans: Optional[str]) -> Optional[str]:
    if not ans:
        return None
    a = ans.strip()
    if len(a) < 20:
        return None
    toks = a.lower().split()
    # low lexical diversity → likely junk
    if len(toks) >= 6 and len(set(toks)) / len(toks) < 0.5:
        return None
    if any(b in a.lower() for b in _BAD_TOKENS):
        return None
    return a

def _local_llm_answer_guarded(prompt: str) -> Optional[str]:
    return _guard(_local_llm_answer(prompt))


# ---------------- Optional OpenAI (only if key present) --------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

def _openai_answer(prompt: str) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    if not api_key or OpenAI is None:
        return None
    try:
        client = OpenAI(api_key=api_key)  # type: ignore[call-arg]
        r = client.chat.completions.create(
            model=model,
            temperature=0.3,
            max_tokens=300,
            messages=[
                {"role": "system", "content": "Be concise and factual."},
                {"role": "user", "content": prompt},
            ],
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return None


# ---------------- Factual detector (to avoid local guesses) -----
_FACTUAL_PAT = re.compile(
    r"\b(who|when|where|which|what)\b.*?\b(president|prime minister|capital|population|date|inauguration|election)\b"
    r"|\b\d{4}\b",
    re.IGNORECASE | re.DOTALL,
)

def _is_factual(t: str) -> bool:
    return bool(_FACTUAL_PAT.search(t))


# ---------------- Filters & helpers --------------------------
WEEKDAYS: Dict[str, int] = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6
}
DAYSETS: Dict[str, list[int]] = {"weekend": [5, 6], "weekdays": [0, 1, 2, 3, 4]}
TIMEWORDS: Dict[str, Tuple[int, int]] = {
    "morning": (6, 11), "afternoon": (12, 17), "evening": (18, 21),
    "night": (22, 23), "late night": (22, 23), "late": (22, 23), "prime time": (18, 23),
}

def _extract_limit(text: str, default: int = 10) -> int:
    m = re.search(r"\btop\s+(\d+)\b", text)
    return int(m.group(1)) if m else default

def _extract_dates(text: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    m = re.search(r"(\d{4}-\d{2}-\d{2}).*?(\d{4}-\d{2}-\d{2})", text)
    if not m:
        return None
    a, b = pd.to_datetime(m.group(1)), pd.to_datetime(m.group(2))
    return (a, b) if a <= b else (b, a)

def _detect_filters(t: str, now: pd.Timestamp) -> Dict[str, Any]:
    """Parse natural language filters → SQL WHERE parts + params + side (pickup/dropoff)."""
    f: Dict[str, Any] = {"where": [], "params": {}, "side": None}

    # Time windows
    if "last month" in t:
        start, end = month_bounds(now, 1)
        f["where"].append("ts >= :start AND ts < :end")
        f["params"].update({"start": start, "end": end})
    elif "this month" in t:
        ref = pd.Timestamp(now).normalize().replace(day=1)
        nxt = (ref + pd.offsets.MonthBegin(1)).normalize()
        f["where"].append("ts >= :start AND ts < :end")
        f["params"].update({"start": ref, "end": nxt})
    elif "last week" in t:
        end = pd.Timestamp(now).normalize()
        start = end - pd.Timedelta(days=7)
        f["where"].append("ts >= :start AND ts < :end")
        f["params"].update({"start": start, "end": end})
    else:
        dr = _extract_dates(t)
        if dr:
            start, end = dr
            end = end + pd.Timedelta(days=1)  # exclusive
            f["where"].append("ts >= :start AND ts < :end")
            f["params"].update({"start": start, "end": end})

    # Weekdays / weekend
    for k, vals in DAYSETS.items():
        if k in t:
            f["where"].append(f"dow IN ({','.join(map(str, vals))})")
    for name, dow in WEEKDAYS.items():
        if name in t:
            f["where"].append(f"dow = {dow}")

    # Hours (keywords + "after/before HH am/pm")
    if "nights" in t and not any(w in t for w in ["morning", "afternoon", "evening", "night "]):
        f["where"].append("hour BETWEEN :h1 AND :h2")
        f["params"].update({"h1": 18, "h2": 23})
    for word, (h1, h2) in TIMEWORDS.items():
        if word in t:
            f["where"].append("hour BETWEEN :h1 AND :h2")
            f["params"].update({"h1": h1, "h2": h2})
    m = re.search(r"after\s+(\d{1,2})\s*(am|pm)?", t)
    if m:
        h = int(m.group(1)) % 12
        if m.group(2) == "pm":
            h += 12
        f["where"].append("hour >= :h_after")
        f["params"]["h_after"] = h
    m = re.search(r"before\s+(\d{1,2})\s*(am|pm)?", t)
    if m:
        h = int(m.group(1)) % 12
        if m.group(2) == "pm":
            h += 12
        f["where"].append("hour < :h_before")
        f["params"]["h_before"] = h

    # Group size
    if re.search(r"\b6\+\b", t) or "large" in t:
        f["where"].append("COALESCE(group_size,0) >= 6")
    m = re.search(r">=\s*(\d+)", t)
    if m:
        f["where"].append("COALESCE(group_size,0) >= :gs")
        f["params"]["gs"] = int(m.group(1))

    # Side: pickups vs drop-offs
    if any(w in t for w in ["pickup", "pick-up", "pickups"]):
        f["side"] = "pickup"
    elif any(w in t for w in ["drop-off", "dropoff", "drop offs"]):
        f["side"] = "dropoff"

    # Venue capture (stop at schedule words) + aliases
    STOP_AHEAD = (
        r"(?:on|after|before|between|from|this|last|week|month|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tonight)"
    )
    m = re.search(r"(?:to|at|near)\s+([a-z0-9 .\'&()/-]+?)(?=\s+" + STOP_AHEAD + r"\b|[?.,]|$)", t)
    if m:
        raw = m.group(1).strip().lower()
        # Aliases
        if "abia" in raw or "bergstrom" in raw or "airport" in raw or re.search(r"\baus\b", raw):
            aliases = ["abia", "airport", "austin-bergstrom", "bergstrom"]
        elif "south congress" in raw or re.search(r"\bs\.?\s*congress\b", raw):
            aliases = ["south congress", "s congress", "s. congress"]
        elif "moody" in raw:
            aliases = ["moody", "moody center"]
        elif "rainey" in raw:
            aliases = ["rainey"]
        elif "domain" in raw:
            aliases = ["domain"]
        else:
            aliases = [raw]

        # Which columns to search
        cols = ["LOWER(dropoff_address)", "LOWER(pickup_address)"]
        if f["side"] == "pickup":
            cols = ["LOWER(pickup_address)"]
        if f["side"] == "dropoff":
            cols = ["LOWER(dropoff_address)"]

        parts = []
        for i, a in enumerate(aliases):
            for j, c in enumerate(cols):
                key = f"venue_{i}_{j}"
                f["params"][key] = f"%{a}%"
                parts.append(f"{c} LIKE :{key}")
        f["where"].append("(" + " OR ".join(parts) + ")")

    return f

def _sql_where(filters: Dict[str, Any]) -> str:
    return "WHERE " + " AND ".join(filters["where"]) if filters["where"] else ""


# ---------------- Intent parser -----------------------------
def parse_intent(text: str) -> Dict[str, Any]:
    t = text.lower().strip()
    intent: Dict[str, Any] = {"action": "fallback", "_raw": t}

    if ("how many" in t or "count" in t) and "last month" in t:
        m = re.search(r"(?:to|went to|at)\s+(.+?)\s+last month", t)
        intent["action"] = "count_trips_last_month_to_venue"
        intent["venue"] = m.group(1).strip() if m else None
        return intent

    if (("when" in t) or ("what time" in t)) and ("downtown" in t) and ("6+" in t or "large" in t or "six" in t):
        intent["action"] = "large_groups_downtown_hours"
        return intent

    if "18" in t and "24" in t:
        intent["action"] = "needs_age_data"
        return intent

    if "top" in t and ("drop" in t or "drop-off" in t or "pickup" in t):
        intent["action"] = "auto_top_locations"
        intent["which"] = "dropoff" if "drop" in t else "pickup"
        intent["limit"] = _extract_limit(t, 10)
        return intent

    if "busiest hour" in t or ("by hour" in t and "trips" in t):
        intent["action"] = "auto_by_hour"
        return intent

    if "average group size" in t or "avg group size" in t:
        intent["action"] = "auto_avg_group_size"
        return intent

    if "count" in t or "how many" in t or "number of trips" in t:
        intent["action"] = "auto_count"
        return intent

    # If it doesn’t look like a data question, route to out-of-scope
    dataish = any(w in t for w in [
        "trip","trips","group","groups","pickup","drop","address","hour","month","weekend","weekday",
        "downtown","moody","domain","rainey","passengers","riders","latitude","longitude"
    ])
    if not dataish:
        intent["action"] = "out_of_scope"
    return intent


# ---------------- Execution engine --------------------------
def execute_intent(con, intent: Dict[str, Any], now: Optional[pd.Timestamp] = None):
    """Return (message: str, df: pandas.DataFrame, debug_sql: Optional[str])."""
    now = now or pd.Timestamp.utcnow()
    action = intent.get("action")
    debug_sql: Optional[str] = None

    if action == "count_trips_last_month_to_venue":
        start, end = month_bounds(now, 1)
        venue = intent.get("venue")
        where_venue = ""
        params: Dict[str, Any] = {"start": start, "end": end}
        if venue:
            where_venue = "AND (LOWER(dropoff_address) LIKE :venue OR LOWER(pickup_address) LIKE :venue)"
            params["venue"] = f"%{str(venue).lower()}%"
        sql = f"""
            SELECT COUNT(*) AS trips
            FROM trips_enriched
            WHERE ts >= :start AND ts < :end
            {where_venue}
        """
        df = run_sql(con, sql, params)
        debug_sql = format_sql_debug(sql, params)
        msg = f"{int(df.iloc[0]['trips'])} groups" + (f" to '{venue}'" if venue else "") + " last month."
        return msg, df, debug_sql

    if action == "large_groups_downtown_hours":
        sql = """
            WITH downtown_trips AS (
                SELECT *
                FROM trips_enriched
                WHERE LOWER(COALESCE(pickup_address,'')) LIKE '%downtown%'
                   OR LOWER(COALESCE(dropoff_address,'')) LIKE '%downtown%'
                   OR (pick_lat_bin BETWEEN 30.25 AND 30.29 AND pick_lon_bin BETWEEN -97.75 AND -97.70)
                   OR (drop_lat_bin BETWEEN 30.25 AND 30.29 AND drop_lon_bin BETWEEN -97.75 AND -97.70)
            )
            SELECT hour, COUNT(*) AS trips
            FROM downtown_trips
            WHERE COALESCE(group_size, 0) >= 6
            GROUP BY hour
            ORDER BY hour
        """
        df = run_sql(con, sql, {})
        debug_sql = sql
        return "Hourly distribution of large (6+) downtown rides:", df, debug_sql

    if action == "needs_age_data":
        return "Age-based questions need the Rider data + Ride Demo files. Upload those to enable 18–24 cohort analysis.", pd.DataFrame(), None

    if action == "auto_top_locations":
        t = intent.get("_raw", "")
        filters = _detect_filters(t, now)
        which = intent.get("which", "dropoff")
        label = "spot"
        lat_bin = "drop_lat_bin" if which == "dropoff" else "pick_lat_bin"
        lon_bin = "drop_lon_bin" if which == "dropoff" else "pick_lon_bin"
        col = "dropoff_address" if which == "dropoff" else "pickup_address"
        sql = f"""
            SELECT
              COALESCE({col}, CONCAT('[', {lat_bin}, ',', {lon_bin}, ']')) AS {label},
              COUNT(*) AS trips
            FROM trips_enriched
            {_sql_where(filters)}
            GROUP BY {label}
            ORDER BY trips DESC
            LIMIT {int(intent.get('limit', 10))}
        """
        df = run_sql(con, sql, filters["params"])
        debug_sql = format_sql_debug(sql, filters["params"])
        return f"Top {intent.get('limit', 10)} {which} spots:", df, debug_sql

    if action == "auto_by_hour":
        t = intent.get("_raw", "")
        filters = _detect_filters(t, now)
        sql = f"""
            SELECT hour, COUNT(*) AS trips
            FROM trips_enriched
            {_sql_where(filters)}
            GROUP BY hour
            ORDER BY hour
        """
        df = run_sql(con, sql, filters["params"])
        debug_sql = format_sql_debug(sql, filters["params"])
        return "Trips by hour:", df, debug_sql

    if action == "auto_avg_group_size":
        t = intent.get("_raw", "")
        filters = _detect_filters(t, now)
        sql = f"""
            SELECT AVG(COALESCE(group_size,0)) AS avg_group_size, COUNT(*) AS trips
            FROM trips_enriched
            {_sql_where(filters)}
        """
        df = run_sql(con, sql, filters["params"])
        debug_sql = format_sql_debug(sql, filters["params"])
        return "Average group size for the requested slice:", df, debug_sql

    if action == "auto_count":
        t = intent.get("_raw", "")
        filters = _detect_filters(t, now)
        sql = f"SELECT COUNT(*) AS trips FROM trips_enriched {_sql_where(filters)}"
        df = run_sql(con, sql, filters["params"])
        debug_sql = format_sql_debug(sql, filters["params"])
        return "Trip count for the requested slice:", df, debug_sql

    if action == "out_of_scope":
        raw_q = str(intent.get("_raw", "")).strip()

        # 1) Brand
        brand = _brand_answer(raw_q)
        if brand:
            return "⚠️ Not from Fetii data / uploaded CSV.\n\n**General answer (company blurb):** " + brand, pd.DataFrame(), None

        # 2) Wikipedia
        wiki = _wiki_answer(raw_q)
        if wiki:
            return "⚠️ Not from Fetii data / uploaded CSV.\n\n**General answer:** " + wiki + "\n\n_Source: Wikipedia_", pd.DataFrame(), None

        # 3) Factual? avoid guessing
        if _is_factual(raw_q):
            msg = ("⚠️ Not from Fetii data / uploaded CSV.\n\n"
                   "I couldn't verify that fact from Wikipedia, so I won't guess. "
                   "Try a Fetii data question like “Top 10 drop-off spots last week”.")
            return msg, pd.DataFrame(), None

        # 4) Local tiny model (guarded)
        local = _local_llm_answer_guarded(raw_q)
        if local:
            return "⚠️ Not from Fetii data / uploaded CSV.\n\n**General answer:** " + local + "\n\n_Source: Local model (FLAN-T5-small)_", pd.DataFrame(), None

        # 5) Optional OpenAI
        gpt = _openai_answer(raw_q)
        if gpt:
            return "⚠️ Not from Fetii data / uploaded CSV.\n\n**General answer:** " + gpt + "\n\n_Source: OpenAI_", pd.DataFrame(), None

        return "⚠️ Not from Fetii data / uploaded CSV.\n\nTry a Fetii data question like: “How many groups went to Moody Center last month?”", pd.DataFrame(), None

    # Fallback: monthly overview
    sql = """
        SELECT year, month, COUNT(*) AS trips, AVG(COALESCE(group_size,0)) AS avg_group_size
        FROM trips_enriched
        GROUP BY year, month
        ORDER BY year, month
        LIMIT 24
    """
    df = run_sql(con, sql, {})
    return "Quick monthly overview:", df, sql

# # ---------------- Optional CLI harness (handy in VS Code) -----------
# if __name__ == "__main__":
#     from data_prep import load_trip_csv_to_duckdb
#     import argparse, os

#     def _default_csv_path() -> str:
#         # preferred filenames/locations
#         candidates = [
#             "FetiiAI_Data_Austin.xlsx - Trip Data.csv",              # project root
#             os.path.join("data", "FetiiAI_Data_Austin.xlsx - Trip Data.csv"),
#             "Trip Data.csv",
#             os.path.join("data", "Trip Data.csv"),
#         ]
#         for c in candidates:
#             p = os.path.abspath(c)
#             if os.path.exists(p):
#                 return p
#         # fall back to the preferred name in the current folder
#         return os.path.abspath("FetiiAI_Data_Austin.xlsx - Trip Data.csv")

#     ap = argparse.ArgumentParser(description="Test NLQ → SQL for Fetii.")
#     ap.add_argument(
#         "--csv",
#         default=_default_csv_path(),
#         help="Path to 'FetiiAI_Data_Austin.xlsx - Trip Data.csv' (auto-detected by default).",
#     )
#     args = ap.parse_args()

#     try:
#         con = load_trip_csv_to_duckdb(args.csv)
#     except Exception as e:
#         print(f"Failed to load CSV at {args.csv}\n{e}")
#         raise SystemExit(1)

#     print(f"Loaded: {args.csv}")
#     print("Ask something (Ctrl+C to exit). Examples:")
#     print(" • How many groups went to Moody Center last month?")
#     print(" • Top 10 drop-off spots last weekend")
#     print(" • When do large groups (6+) typically ride downtown?")
#     print(" • Average group size near Rainey on weekends")
#     while True:
#         try:
#             q = input("\nQ: ").strip()
#             if not q:
#                 continue
#             msg, df, sql_dbg = execute_intent(con, parse_intent(q))
#             print("\n" + msg)
#             if not df.empty:
#                 print(df.head(20).to_string(index=False))
#             if sql_dbg:
#                 print("\nSQL:\n" + sql_dbg)
#         except KeyboardInterrupt:
#             print("\nbye!")
#             break
        
# ---------------- Optional CLI harness (handy in VS Code) -----------
if __name__ == "__main__":
    from data_prep import load_trip_csv_to_duckdb
    import argparse, os, sys

    def _default_csv_path() -> str:
        candidates = [
            "FetiiAI_Data_Austin.xlsx - Trip Data.csv",
            os.path.join("data", "FetiiAI_Data_Austin.xlsx - Trip Data.csv"),
            "Trip Data.csv",
            os.path.join("data", "Trip Data.csv"),
        ]
        for c in candidates:
            p = os.path.abspath(c)
            if os.path.exists(p):
                return p
        return os.path.abspath("FetiiAI_Data_Austin.xlsx - Trip Data.csv")

    ap = argparse.ArgumentParser(description="Test NLQ → SQL for Fetii.")
    ap.add_argument(
        "--csv",
        default=_default_csv_path(),
        help="Path to 'FetiiAI_Data_Austin.xlsx - Trip Data.csv' (auto-detected by default).",
    )
    args = ap.parse_args()

    try:
        con = load_trip_csv_to_duckdb(args.csv)
    except Exception as e:
        print(f"Failed to load CSV at {args.csv}\n{e}")
        sys.exit(1)

    EXAMPLES = [
        "How many groups went to Moody Center last month?",
        "Top 10 drop-off spots last weekend",
        "When do large groups (6+) typically ride downtown?",
        "Average group size near Rainey on weekends",
    ]

    print(f"Loaded: {args.csv}")
    print("Type a question, or 'help' for examples, or 'exit' to quit.")

    while True:
        try:
            q = input("\nQ: ").strip()
            if not q:
                continue

            cmd = q.lower()
            if cmd in {"exit", "quit", "bye", "q"}:
                print("bye!")
                break
            if cmd in {"help", "examples"}:
                print("\nExamples:")
                for e in EXAMPLES:
                    print(" • " + e)
                continue

            msg, df, sql_dbg = execute_intent(con, parse_intent(q))
            print("\n" + msg)
            if df is not None and not df.empty:
                print(df.head(20).to_string(index=False))
            if sql_dbg:
                print("\nSQL:\n" + sql_dbg)

        except (KeyboardInterrupt, EOFError):
            print("\nbye!")
            break
