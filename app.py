# from __future__ import annotations
# import os, io, tempfile
# import pandas as pd
# import streamlit as st
# import pydeck as pdk

# from data_prep import autoload_connection, load_trip_csv_to_duckdb
# from nlq import parse_intent, execute_intent

# # ------------------ Page ------------------
# st.set_page_config(page_title="Fetii Trips — Austin Chat", layout="wide")

# # ------------------ Themes -----------------
# DARK_CSS = """
# :root { --bg:#0b0f19; --panel:#0f1426; --muted:#8c96b2; --text:#eaf0ff; --border:#1b2240; --chip:#1b2240; --accent:#6ea8ff; }
# html, body, [class*="stApp"]{ background:var(--bg); color:var(--text); }
# h1,h2,h3,h4{ color:var(--text); }
# .answer-card,.kpi,.panel{ background:var(--panel); border:1px solid var(--border); border-radius:14px; }
# .kpi{ padding:12px 14px; } .kpi .label{font-size:11px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase}
# .kpi .value{font-size:20px;font-weight:700;margin-top:4px;color:var(--text)}
# .stChatMessage{ background:transparent }
# .stButton > button{ background:var(--panel); color:var(--text); border:1px solid var(--border); border-radius:16px; padding:10px 14px; }
# .stButton > button:hover{ border-color:var(--accent); }
# """

# LIGHT_CSS = """
# :root { --bg:#f7fafc; --panel:#ffffff; --muted:#64748b; --text:#0b1220; --border:#e6e8ee; --chip:#eef2ff; --accent:#1d4ed8; }
# html, body, [class*="stApp"]{ background:var(--bg); color:var(--text); }
# h1,h2,h3,h4{ color:var(--text); }
# .answer-card,.kpi,.panel{ background:var(--panel); border:1px solid var(--border); border-radius:14px; }
# .kpi{ padding:12px 14px; } .kpi .label{font-size:11px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase}
# .kpi .value{font-size:20px;font-weight:700;margin-top:4px;color:var(--text)}
# .stChatMessage{ background:transparent }
# .stButton > button{ background:var(--panel); color:var(--text); border:1px solid var(--border); border-radius:16px; padding:10px 14px; }
# .stButton > button:hover{ border-color:var(--accent); }
# """

# # theme toggle
# if "ui_theme" not in st.session_state:
#     st.session_state.ui_theme = "dark"
# theme_is_dark = st.sidebar.toggle("🌙 Dark mode", value=(st.session_state.ui_theme == "dark"))
# st.session_state.ui_theme = "dark" if theme_is_dark else "light"
# st.markdown(f"<style>{DARK_CSS if theme_is_dark else LIGHT_CSS}</style>", unsafe_allow_html=True)

# # Map provider/style (no API key)
# pdk.settings.map_provider = "carto"
# pdk.settings.map_style = "dark" if theme_is_dark else "light"
# DOT_COLOR = [255, 210, 30] if theme_is_dark else [0, 90, 181]  # tiny dots: amber on dark, blue on light

# # bin precision for dedupe (4 ≈ 11 m)
# BIN_PRECISION = 4

# # ------------------ Sidebar ------------------
# with st.sidebar:
#     st.markdown("### 🚐 Fetii Trips")
#     with st.expander("⚙️ Data source (optional)", expanded=False):
#         st.write("Using the packaged dataset by default. Upload a different Trip CSV if you want.")
#         up = st.file_uploader("Upload Trip Data CSV", type=["csv"], accept_multiple_files=False, label_visibility="collapsed")
#         if up:
#             tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
#             tmp.write(up.read()); tmp.flush()
#             st.session_state["upload_path"] = tmp.name
#             st.success("Uploaded. Reloading…")
#             st.rerun()

# @st.cache_resource(show_spinner=False)
# def get_connection(upload_path: str | None):
#     if upload_path and os.path.exists(upload_path):
#         return load_trip_csv_to_duckdb(upload_path)
#     return autoload_connection()

# try:
#     con = get_connection(st.session_state.get("upload_path"))
# except Exception as e:
#     con = None
#     st.sidebar.error(f"Dataset error:\n{e}")

# # ------------------ Header / KPIs ------------------
# st.markdown("## Fetii Trips")
# st.caption("Ask natural-language questions about trips. (Age-based queries need Rider + Demo files.)")

# if con:
#     meta = con.execute("""
#         SELECT COUNT(*) trips,
#                ROUND(AVG(COALESCE(group_size,0)),2) avg_group
#         FROM trips
#     """).df().iloc[0]
#     k1,k2,k3,k4 = st.columns(4)
#     with k1: st.markdown(f'<div class="kpi"><div class="label">Trips</div><div class="value">{int(meta["trips"])}</div></div>', unsafe_allow_html=True)
#     with k2: st.markdown(f'<div class="kpi"><div class="label">Avg Group</div><div class="value">{meta["avg_group"]}</div></div>', unsafe_allow_html=True)
#     with k3:
#         try:
#             hr = con.execute("SELECT hour FROM (SELECT hour, COUNT(*) c FROM trips_enriched GROUP BY hour ORDER BY c DESC LIMIT 1)").df().iloc[0,0]
#         except Exception: hr = "—"
#         st.markdown(f'<div class="kpi"><div class="label">Busiest Hour</div><div class="value">{hr}</div></div>', unsafe_allow_html=True)
#     with k4:
#         try:
#             spot = con.execute("""
#                 WITH t AS (
#                   SELECT COALESCE(dropoff_address, CONCAT('[', drop_lat_bin, ',', drop_lon_bin, ']')) AS spot,
#                          COUNT(*) c
#                   FROM trips_enriched GROUP BY 1 ORDER BY c DESC LIMIT 1)
#                 SELECT spot FROM t
#             """).df().iloc[0,0]
#         except Exception: spot = "—"
#         st.markdown(f'<div class="kpi"><div class="label">Top Drop-off</div><div class="value">{spot}</div></div>', unsafe_allow_html=True)

# st.markdown("")

# # ------------------ Suggestion chips (auto-run) ------------------
# def _queue(q: str):
#     st.session_state["queued_q"] = q
#     st.session_state["prefill"] = q

# c1,c2,c3,c4 = st.columns(4)
# with c1:
#     if st.button("Moody last month"): _queue("How many groups went to Moody Center last month?")
# with c2:
#     if st.button("Top 10 Rainey (weekend)"): _queue("Top 10 drop-off spots near Rainey on weekend")
# with c3:
#     if st.button("Large 6+ downtown (when?)"): _queue("When do large groups (6+) typically ride downtown?")
# with c4:
#     if st.button("Trips by hour (Saturday)"): _queue("Trips by hour on Saturday")

# # ------------------ Chat history ------------------
# if "messages" not in st.session_state:
#     st.session_state.messages = []
#     st.session_state.messages.append({
#         "role":"assistant",
#         "text":"Hi! Ask me about Fetii trips, or tap a chip above.\n\n"
#                "• How many groups went to Moody Center last month?\n"
#                "• Top 10 drop-off spots last weekend\n"
#                "• When do large groups (6+) ride downtown?\n"
#                "• Average group size near Rainey on weekends",
#         "df":None,"sql":None
#     })

# def _pack_df(df: pd.DataFrame | None):
#     if df is None or df.empty: return None
#     return {"columns":list(df.columns), "data":df.to_dict(orient="records")}
# def _unpack_df(blob):
#     if not blob: return None
#     return pd.DataFrame(blob["data"], columns=blob["columns"])
# def _df_csv_bytes(df: pd.DataFrame)->bytes:
#     buf = io.StringIO(); df.to_csv(buf, index=False); return buf.getvalue().encode("utf-8")

# # render history
# for m in st.session_state.messages:
#     with st.chat_message(m["role"], avatar="🚐" if m["role"]=="assistant" else "🧑"):
#         st.markdown(m["text"])
#         df = _unpack_df(m["df"])
#         if df is not None and not df.empty:
#             st.dataframe(df, use_container_width=True)
#             if {"hour","trips"}.issubset(df.columns):
#                 st.bar_chart(df.set_index("hour")["trips"])
#         if m.get("sql"):
#             with st.expander("Show SQL", expanded=False):
#                 st.code(m["sql"], language="sql")

# # ------------------ Chat input (autorun queued chip; ensure box returns) ------------------
# default_q = st.session_state.pop("prefill", "How many groups went to Moody Center last month?")
# queued = st.session_state.pop("queued_q", None)
# used_queued = queued is not None
# user_q = queued if used_queued else st.chat_input(placeholder=default_q)

# if user_q:
#     st.session_state.messages.append({"role":"user","text":user_q,"df":None,"sql":None})
#     with st.chat_message("user", avatar="🧑"):
#         st.markdown(user_q)

#     if not con:
#         with st.chat_message("assistant", avatar="🚐"):
#             st.warning("I can’t find the dataset. Keep the CSV named **FetiiAI_Data_Austin.xlsx - Trip Data.csv** "
#                        "in the project root (or in `./data/`). Or upload a CSV from the sidebar expander.")
#     else:
#         intent = parse_intent(user_q)
#         msg, df, debug_sql = execute_intent(con, intent)

#         with st.chat_message("assistant", avatar="🚐"):
#             if msg.strip().startswith("⚠️"): st.warning(msg)
#             else: st.markdown(f'<div class="answer-card">{msg}</div>', unsafe_allow_html=True)

#             if df is not None and not df.empty:
#                 st.dataframe(df, use_container_width=True)
#                 if {"hour","trips"}.issubset(df.columns):
#                     st.bar_chart(df.set_index("hour")["trips"])

#                 # ---- Tiny, deduped dots map (one dot per location; hover shows aggregate) ----
#                 try:
#                     agg = con.execute(f"""
#                         SELECT
#                           ROUND(CAST(dropoff_latitude  AS DOUBLE), {BIN_PRECISION}) AS lat,
#                           ROUND(CAST(dropoff_longitude AS DOUBLE), {BIN_PRECISION}) AS lon,
#                           COUNT(*) AS trips
#                         FROM trips_enriched
#                         WHERE dropoff_latitude IS NOT NULL AND dropoff_longitude IS NOT NULL
#                         GROUP BY 1,2
#                         HAVING COUNT(*) > 0
#                         ORDER BY trips DESC
#                         LIMIT 2000
#                     """).df()

#                     if not agg.empty:
#                         st.markdown("**Hotspots (drop-off, deduped)**")
#                         layer = pdk.Layer(
#                             "ScatterplotLayer",
#                             data=agg,
#                             get_position='[lon, lat]',
#                             get_fill_color=DOT_COLOR,
#                             get_radius=2,             # base radius (meters)
#                             radius_scale=1,
#                             radius_min_pixels=1,      # keep dots tiny
#                             radius_max_pixels=2,
#                             pickable=True,
#                         )
#                         deck = pdk.Deck(
#                             initial_view_state=pdk.ViewState(latitude=30.2672, longitude=-97.7431, zoom=11),
#                             layers=[layer],
#                             tooltip={"html":"<b>Trips:</b> {trips}<br/><b>Lat:</b> {lat}<br/><b>Lon:</b> {lon}"}
#                         )
#                         st.pydeck_chart(deck, use_container_width=True)
#                 except Exception:
#                     pass

#                 st.download_button("Download results (CSV)", _df_csv_bytes(df), "fetii_results.csv", "text/csv")

#             if debug_sql:
#                 with st.expander("Show SQL"):
#                     st.code(debug_sql, language="sql")

#         st.session_state.messages.append({"role":"assistant","text":msg,"df":_pack_df(df),"sql":debug_sql})

#     if used_queued:
#         st.rerun()

# app.py  — Streamlit main entry (Cloud-ready)
from __future__ import annotations
import os, io, tempfile
import pandas as pd
import streamlit as st
import pydeck as pdk

try:
    from data_prep import autoload_connection, load_trip_csv_to_duckdb
    from nlq import parse_intent, execute_intent
    DATA_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    DATA_AVAILABLE = False
    # Create dummy functions to prevent further errors
    def autoload_connection(*args, **kwargs):
        raise ImportError("Data modules not available")
    def load_trip_csv_to_duckdb(*args, **kwargs):
        raise ImportError("Data modules not available")
    def parse_intent(*args, **kwargs):
        return {"action": "error", "message": "Data modules not available"}
    def execute_intent(*args, **kwargs):
        return "Data modules not available", None, None

# ------------------ Page ------------------
st.set_page_config(page_title="Fetii Trips — Austin Chat", layout="wide")

# ------------------ Themes -----------------
DARK_CSS = """
:root { --bg:#0b0f19; --panel:#0f1426; --muted:#8c96b2; --text:#eaf0ff; --border:#1b2240; --chip:#1b2240; --accent:#6ea8ff; }
html, body, [class*="stApp"]{ background:var(--bg); color:var(--text); }
h1,h2,h3,h4{ color:var(--text); }
.answer-card,.kpi,.panel{ background:var(--panel); border:1px solid var(--border); border-radius:14px; }
.kpi{ padding:12px 14px; } .kpi .label{font-size:11px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase}
.kpi .value{font-size:20px;font-weight:700;margin-top:4px;color:var(--text)}
.stChatMessage{ background:transparent }
.stButton > button{ background:var(--panel); color:var(--text); border:1px solid var(--border); border-radius:16px; padding:10px 14px; }
.stButton > button:hover{ border-color:var(--accent); }
"""

LIGHT_CSS = """
:root { --bg:#f7fafc; --panel:#ffffff; --muted:#64748b; --text:#0b1220; --border:#e6e8ee; --chip:#eef2ff; --accent:#1d4ed8; }
html, body, [class*="stApp"]{ background:var(--bg); color:var(--text); }
h1,h2,h3,h4{ color:var(--text); }
.answer-card,.kpi,.panel{ background:var(--panel); border:1px solid var(--border); border-radius:14px; }
.kpi{ padding:12px 14px; } .kpi .label{font-size:11px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase}
.kpi .value{font-size:20px;font-weight:700;margin-top:4px;color:var(--text)}
.stChatMessage{ background:transparent }
.stButton > button{ background:var(--panel); color:var(--text); border:1px solid var(--border); border-radius:16px; padding:10px 14px; }
.stButton > button:hover{ border-color:var(--accent); }
"""

# theme toggle
if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "dark"
theme_is_dark = st.sidebar.toggle("🌙 Dark mode", value=(st.session_state.ui_theme == "dark"))
st.session_state.ui_theme = "dark" if theme_is_dark else "light"
st.markdown(f"<style>{DARK_CSS if theme_is_dark else LIGHT_CSS}</style>", unsafe_allow_html=True)

# Map provider/style (no API key)
pdk.settings.map_provider = "carto"
pdk.settings.map_style = "dark" if theme_is_dark else "light"
DOT_COLOR = [255, 210, 30] if theme_is_dark else [0, 90, 181]  # tiny dots: amber on dark, blue on light

# bin precision for dedupe (4 ≈ 11 m)
BIN_PRECISION = 4

# ------------------ Sidebar ------------------
with st.sidebar:
    st.markdown("### 🚐 Fetii Trips")
    with st.expander("⚙️ Data source (optional)", expanded=False):
        st.write("Using the packaged dataset by default. Upload a different Trip CSV if you want.")
        up = st.file_uploader("Upload Trip Data CSV", type=["csv"], accept_multiple_files=False, label_visibility="collapsed")
        if up:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.write(up.read())
            tmp.flush()
            tmp.close()  # <-- ensure the handle is closed (Windows/Cloud safe)
            st.session_state["upload_path"] = tmp.name
            st.success("Uploaded. Reloading…")
            st.rerun()

@st.cache_resource(show_spinner=False)
def get_connection(upload_path: str | None):
    if upload_path and os.path.exists(upload_path):
        return load_trip_csv_to_duckdb(upload_path)
    return autoload_connection()   # loads data/trips.csv (or other supported names)

if not DATA_AVAILABLE:
    con = None
    st.sidebar.error("Required data modules are not available. Please check your deployment configuration.")
else:
    try:
        con = get_connection(st.session_state.get("upload_path"))
    except Exception as e:
        con = None
        st.sidebar.error(f"Dataset error:\n{e}")

# ------------------ Header / KPIs ------------------
st.markdown("## Fetii Trips")
st.caption("Ask natural-language questions about trips. (Age-based queries need Rider + Demo files.)")

if con:
    meta = con.execute("""
        SELECT COUNT(*) trips,
               ROUND(AVG(COALESCE(group_size,0)),2) avg_group
        FROM trips
    """).df().iloc[0]
    k1,k2,k3,k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi"><div class="label">Trips</div><div class="value">{int(meta["trips"])}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi"><div class="label">Avg Group</div><div class="value">{meta["avg_group"]}</div></div>', unsafe_allow_html=True)
    with k3:
        try:
            hr = con.execute("""
                SELECT hour FROM (
                    SELECT hour, COUNT(*) c
                    FROM trips_enriched
                    GROUP BY hour
                    ORDER BY c DESC
                    LIMIT 1
                )
            """).df().iloc[0,0]
        except Exception:
            hr = "—"
        st.markdown(f'<div class="kpi"><div class="label">Busiest Hour</div><div class="value">{hr}</div></div>', unsafe_allow_html=True)
    with k4:
        try:
            spot = con.execute("""
                WITH t AS (
                  SELECT COALESCE(dropoff_address, CONCAT('[', drop_lat_bin, ',', drop_lon_bin, ']')) AS spot,
                         COUNT(*) c
                  FROM trips_enriched GROUP BY 1 ORDER BY c DESC LIMIT 1
                )
                SELECT spot FROM t
            """).df().iloc[0,0]
        except Exception:
            spot = "—"
        st.markdown(f'<div class="kpi"><div class="label">Top Drop-off</div><div class="value">{spot}</div></div>', unsafe_allow_html=True)

st.markdown("")

# ------------------ Suggestion chips (auto-run) ------------------
def _queue(q: str):
    st.session_state["queued_q"] = q
    st.session_state["prefill"] = q

c1,c2,c3,c4 = st.columns(4)
with c1:
    if st.button("Moody last month"): _queue("How many groups went to Moody Center last month?")
with c2:
    if st.button("Top 10 Rainey (weekend)"): _queue("Top 10 drop-off spots near Rainey on weekend")
with c3:
    if st.button("Large 6+ downtown (when?)"): _queue("When do large groups (6+) typically ride downtown?")
with c4:
    if st.button("Trips by hour (Saturday)"): _queue("Trips by hour on Saturday")

# ------------------ Chat history ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role":"assistant",
        "text":"Hi! Ask me about Fetii trips, or tap a chip above.\n\n"
               "• How many groups went to Moody Center last month?\n"
               "• Top 10 drop-off spots last weekend\n"
               "• When do large groups (6+) ride downtown?\n"
               "• Average group size near Rainey on weekends",
        "html": None,
        "df":None,"sql":None
    })

def _pack_df(df: pd.DataFrame | None):
    if df is None or df.empty: return None
    return {"columns":list(df.columns), "data":df.to_dict(orient="records")}
def _unpack_df(blob):
    if not blob: return None
    return pd.DataFrame(blob["data"], columns=blob["columns"])
def _df_csv_bytes(df: pd.DataFrame)->bytes:
    buf = io.StringIO(); df.to_csv(buf, index=False); return buf.getvalue().encode("utf-8")

# render history (prefer HTML if present → keeps answer-card styling)
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🚐" if m["role"]=="assistant" else "🧑"):
        if m.get("html"):
            st.markdown(m["html"], unsafe_allow_html=True)
        else:
            st.markdown(m["text"])
        df = _unpack_df(m["df"])
        if df is not None and not df.empty:
            st.dataframe(df, use_container_width=True)
            if {"hour","trips"}.issubset(df.columns):
                st.bar_chart(df.set_index("hour")["trips"])
        if m.get("sql"):
            with st.expander("Show SQL", expanded=False):
                st.code(m["sql"], language="sql")

# ------------------ Chat input (autorun queued chip; ensure box returns) ------------------
default_q = st.session_state.pop("prefill", "How many groups went to Moody Center last month?")
queued = st.session_state.pop("queued_q", None)
used_queued = queued is not None
user_q = queued if used_queued else st.chat_input(placeholder=default_q)

if user_q:
    st.session_state.messages.append({"role":"user","text":user_q,"html":None,"df":None,"sql":None})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_q)

    if not con:
        with st.chat_message("assistant", avatar="🚐"):
            st.warning("I can’t find the dataset. Keep the CSV named **FetiiAI_Data_Austin.xlsx - Trip Data.csv** "
                       "in the project root (or in `./data/`). Or upload a CSV from the sidebar expander.")
    else:
        intent = parse_intent(user_q)
        msg, df, debug_sql = execute_intent(con, intent)

        with st.chat_message("assistant", avatar="🚐"):
            if msg.strip().startswith("⚠️"):
                st.warning(msg)
                card_html = None
            else:
                card_html = f'<div class="answer-card">{msg}</div>'
                st.markdown(card_html, unsafe_allow_html=True)

            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True)
                if {"hour","trips"}.issubset(df.columns):
                    st.bar_chart(df.set_index("hour")["trips"])

                # ---- Tiny, deduped dots map (one dot per location; hover shows aggregate) ----
                try:
                    agg = con.execute(f"""
                        SELECT
                          ROUND(CAST(dropoff_latitude  AS DOUBLE), {BIN_PRECISION}) AS lat,
                          ROUND(CAST(dropoff_longitude AS DOUBLE), {BIN_PRECISION}) AS lon,
                          COUNT(*) AS trips
                        FROM trips_enriched
                        WHERE dropoff_latitude IS NOT NULL AND dropoff_longitude IS NOT NULL
                        GROUP BY 1,2
                        HAVING COUNT(*) > 0
                        ORDER BY trips DESC
                        LIMIT 2000
                    """).df()

                    if not agg.empty:
                        st.markdown("**Hotspots (drop-off, deduped)**")
                        layer = pdk.Layer(
                            "ScatterplotLayer",
                            data=agg,
                            get_position='[lon, lat]',
                            get_fill_color=DOT_COLOR,
                            get_radius=2,
                            radius_scale=1,
                            radius_min_pixels=1,
                            radius_max_pixels=2,
                            pickable=True,
                        )
                        deck = pdk.Deck(
                            initial_view_state=pdk.ViewState(latitude=30.2672, longitude=-97.7431, zoom=11),
                            layers=[layer],
                            tooltip={"html":"<b>Trips:</b> {trips}<br/><b>Lat:</b> {lat}<br/><b>Lon:</b> {lon}"}
                        )
                        st.pydeck_chart(deck, use_container_width=True)
                except Exception:
                    pass

                st.download_button("Download results (CSV)", _df_csv_bytes(df), "fetii_results.csv", "text/csv")

            if debug_sql:
                with st.expander("Show SQL"):
                    st.code(debug_sql, language="sql")

        # persist in history (store html so the card styling survives reruns)
        st.session_state.messages.append({
            "role":"assistant",
            "text":msg,
            "html":card_html,
            "df":_pack_df(df),
            "sql":debug_sql
        })

    if used_queued:
        st.rerun()
