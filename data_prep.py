# # data_prep.py — Load & prepare Fetii trip data into DuckDB (VS Code)
# # -------------------------------------------------------------------
# # Usage from code:
# #   from data_prep import load_trip_csv_to_duckdb, autoload_connection
# #   con = autoload_connection()  # auto-detects the CSV in ./ or ./data/
# #   # or:
# #   con = load_trip_csv_to_duckdb("FetiiAI_Data_Austin.xlsx - Trip Data.csv")

# from __future__ import annotations
# import os
# import duckdb
# import pandas as pd
# from typing import Optional

# TRIP_COL_MAP = {
#     "Trip ID": "trip_id",
#     "Booking User ID": "booking_user_id",
#     "Pick Up Latitude": "pickup_latitude",
#     "Pick Up Longitude": "pickup_longitude",
#     "Drop Off Latitude": "dropoff_latitude",
#     "Drop Off Longitude": "dropoff_longitude",
#     "Pick Up Address": "pickup_address",
#     "Drop Off Address": "dropoff_address",
#     "Trip Date and Time": "timestamp",
#     "Total Passengers": "group_size",
# }

# def _normalize_trip_df(df: pd.DataFrame) -> pd.DataFrame:
#     """Rename columns, coerce types, trim strings."""
#     df = df.rename(columns=TRIP_COL_MAP)

#     # ensure all required normalized columns exist
#     for need in TRIP_COL_MAP.values():
#         if need not in df.columns:
#             df[need] = pd.NA

#     # timestamp
#     if "timestamp" in df.columns:
#         df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

#     # numerics
#     for c in ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude", "group_size"]:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors="coerce")

#     # text cleanup
#     for c in ["pickup_address", "dropoff_address"]:
#         if c in df.columns:
#             df[c] = df[c].fillna("").astype(str).str.strip()

#     return df

# def _create_views(con: duckdb.DuckDBPyConnection) -> None:
#     """Create enriched view and placeholder age view."""
#     con.execute(
#         """
#         CREATE OR REPLACE VIEW trips_enriched AS
#         SELECT
#             *,
#             CAST(timestamp AS TIMESTAMP)                               AS ts,
#             EXTRACT(YEAR  FROM timestamp)                              AS year,
#             EXTRACT(MONTH FROM timestamp)                              AS month,
#             CAST(STRFTIME(timestamp, '%w') AS INTEGER)                 AS dow,   -- 0=Sun..6=Sat
#             EXTRACT(HOUR  FROM timestamp)                              AS hour,
#             CASE WHEN STRFTIME(timestamp, '%w') IN ('0','6') THEN TRUE ELSE FALSE END AS is_weekend,
#             ROUND(CAST(pickup_latitude   AS DOUBLE),  3)               AS pick_lat_bin,
#             ROUND(CAST(pickup_longitude  AS DOUBLE),  3)               AS pick_lon_bin,
#             ROUND(CAST(dropoff_latitude  AS DOUBLE),  3)               AS drop_lat_bin,
#             ROUND(CAST(dropoff_longitude AS DOUBLE),  3)               AS drop_lon_bin
#         FROM trips;
#         """
#     )
#     # placeholder so age-based queries fail gracefully
#     con.execute("CREATE OR REPLACE VIEW trip_age_flags AS SELECT 1 WHERE 1=0;")

# def load_trip_csv_to_duckdb(csv_path: str, database_path: Optional[str] = ":memory:") -> duckdb.DuckDBPyConnection:
#     """
#     Load the Trip CSV into DuckDB and return a connection.
#     Set database_path to a filename (e.g. 'fetii.duckdb') if you want persistence.
#     """
#     if not os.path.exists(csv_path):
#         raise FileNotFoundError(f"CSV not found: {csv_path}")

#     try:
#         trips = pd.read_csv(csv_path)
#     except UnicodeDecodeError:
#         trips = pd.read_csv(csv_path, encoding="utf-8", errors="ignore")

#     trips = _normalize_trip_df(trips)

#     con = duckdb.connect(database=database_path or ":memory:")
#     con.register("trips_df", trips)
#     con.execute("CREATE TABLE trips AS SELECT * FROM trips_df;")
#     _create_views(con)
#     return con

# def autoload_connection(database_path: Optional[str] = ":memory:") -> duckdb.DuckDBPyConnection:
#     """
#     Try to find the CSV automatically in common locations and load it.
#     Looks for your exact file name from the hackathon:
#       - ./FetiiAI_Data_Austin.xlsx - Trip Data.csv
#       - ./data/FetiiAI_Data_Austin.xlsx - Trip Data.csv
#       - ./Trip Data.csv
#       - ./data/Trip Data.csv
#     """
#     candidates = [
#         "FetiiAI_Data_Austin.xlsx - Trip Data.csv",
#         os.path.join("data", "FetiiAI_Data_Austin.xlsx - Trip Data.csv"),
#         "Trip Data.csv",
#         os.path.join("data", "Trip Data.csv"),
#     ]
#     for c in candidates:
#         p = os.path.abspath(c)
#         if os.path.exists(p):
#             return load_trip_csv_to_duckdb(p, database_path=database_path)
#     raise FileNotFoundError(
#         "Could not auto-detect the trip CSV. Put your file at one of:\n"
#         "  ./FetiiAI_Data_Austin.xlsx - Trip Data.csv\n"
#         "  ./data/FetiiAI_Data_Austin.xlsx - Trip Data.csv\n"
#         "  ./Trip Data.csv\n"
#         "  ./data/Trip Data.csv"
#     )

# # --- Optional CLI for quick checks in VS Code terminal ---
# if __name__ == "__main__":
#     import argparse
#     ap = argparse.ArgumentParser(description="Load Fetii Trip CSV into DuckDB")
#     ap.add_argument("--csv", default=None, help="Path to the Trip CSV (auto-detect if omitted)")
#     ap.add_argument("--db", default=":memory:", help="DuckDB path (':memory:' or e.g. fetii.duckdb)")
#     args = ap.parse_args()

#     if args.csv:
#         con = load_trip_csv_to_duckdb(args.csv, database_path=args.db)
#         src = args.csv
#     else:
#         con = autoload_connection(database_path=args.db)
#         src = "(auto-detected)"

#     print(f"Loaded {src} → DuckDB [{args.db}]")
#     print(con.execute("SELECT COUNT(*) trips, ROUND(AVG(COALESCE(group_size,0)),2) avg_group FROM trips").df())
# data_prep.py
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from typing import Optional

# Try to import duckdb with fallback
try:
    import duckdb
except ImportError as e:
    print(f"Warning: DuckDB not available: {e}")
    print("Please install duckdb: pip install duckdb")
    duckdb = None

TRIP_COL_MAP = {
    "Trip ID": "trip_id",
    "Booking User ID": "booking_user_id",
    "Pick Up Latitude": "pickup_latitude",
    "Pick Up Longitude": "pickup_longitude",
    "Drop Off Latitude": "dropoff_latitude",
    "Drop Off Longitude": "dropoff_longitude",
    "Pick Up Address": "pickup_address",
    "Drop Off Address": "dropoff_address",
    "Trip Date and Time": "timestamp",
    "Total Passengers": "group_size",
}

def _normalize_trip_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=TRIP_COL_MAP)
    for need in TRIP_COL_MAP.values():
        if need not in df.columns:
            df[need] = pd.NA
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude","group_size"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["pickup_address","dropoff_address"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()
    return df

def _create_views(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE OR REPLACE VIEW trips_enriched AS
        SELECT
            *,
            CAST(timestamp AS TIMESTAMP)                               AS ts,
            EXTRACT(YEAR  FROM timestamp)                              AS year,
            EXTRACT(MONTH FROM timestamp)                              AS month,
            CAST(STRFTIME(timestamp, '%w') AS INTEGER)                 AS dow,   -- 0=Sun..6=Sat
            EXTRACT(HOUR  FROM timestamp)                              AS hour,
            CASE WHEN STRFTIME(timestamp, '%w') IN ('0','6') THEN TRUE ELSE FALSE END AS is_weekend,
            ROUND(CAST(pickup_latitude   AS DOUBLE),  3)               AS pick_lat_bin,
            ROUND(CAST(pickup_longitude  AS DOUBLE),  3)               AS pick_lon_bin,
            ROUND(CAST(dropoff_latitude  AS DOUBLE),  3)               AS drop_lat_bin,
            ROUND(CAST(dropoff_longitude AS DOUBLE),  3)               AS drop_lon_bin
        FROM trips;
    """)
    con.execute("CREATE OR REPLACE VIEW trip_age_flags AS SELECT 1 WHERE 1=0;")

def load_trip_csv_to_duckdb(csv_path: str, database_path: Optional[str] = ":memory:"):
    if duckdb is None:
        raise ImportError("DuckDB is not available. Please install it with: pip install duckdb")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    try:
        trips = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        trips = pd.read_csv(csv_path, encoding="utf-8", errors="ignore")
    trips = _normalize_trip_df(trips)

    con = duckdb.connect(database=database_path or ":memory:")
    con.register("trips_df", trips)
    con.execute("CREATE TABLE trips AS SELECT * FROM trips_df;")
    _create_views(con)
    return con

def autoload_connection(database_path: Optional[str] = ":memory:"):
    """
    Try common repo locations; prefer clean `data/trips.csv` for cloud deploys.
    """
    if duckdb is None:
        raise ImportError("DuckDB is not available. Please install it with: pip install duckdb")
    
    root = Path(__file__).parent
    candidates = [
        root / "data" / "trips.csv",
        root / "trips.csv",
        root / "FetiiAI_Data_Austin.xlsx - Trip Data.csv",
        root / "data" / "FetiiAI_Data_Austin.xlsx - Trip Data.csv",
        root / "Trip Data.csv",
        root / "data" / "Trip Data.csv",
    ]
    for p in candidates:
        if p.exists():
            return load_trip_csv_to_duckdb(str(p), database_path=database_path)
    raise FileNotFoundError(
        "Could not auto-detect the trip CSV. Place a copy at one of:\n"
        "  data/trips.csv   (recommended)\n"
        "  trips.csv\n"
        "  FetiiAI_Data_Austin.xlsx - Trip Data.csv\n"
        "  data/FetiiAI_Data_Austin.xlsx - Trip Data.csv\n"
        "  Trip Data.csv\n"
        "  data/Trip Data.csv"
    )

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Load Fetii Trip CSV into DuckDB")
    ap.add_argument("--csv", default=None, help="Path to the Trip CSV (auto-detect if omitted)")
    ap.add_argument("--db", default=":memory:", help="DuckDB path (':memory:' or e.g. fetii.duckdb)")
    args = ap.parse_args()

    if args.csv:
        con = load_trip_csv_to_duckdb(args.csv, database_path=args.db)
        src = args.csv
    else:
        con = autoload_connection(database_path=args.db)
        src = "(auto-detected)"
    print(f"Loaded {src} → DuckDB [{args.db}]")
    print(con.execute("SELECT COUNT(*) trips, ROUND(AVG(COALESCE(group_size,0)),2) avg_group FROM trips").df())
