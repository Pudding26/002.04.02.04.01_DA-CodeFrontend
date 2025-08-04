"""
02_Visualization.py

Multipage Streamlit app page: Data Viewer Dashboard.

Responsibilities:
  - Manage central session state
  - Render sidebar (data source, filters, visualization choice)
  - Fetch data from backend
  - Dispatch visualization rendering
  - Build and display shareable state links (?state=...)
"""

import os, sys
import pandas as pd
import streamlit as st
import requests

from utils.hosts import BASE_URL_VISU
sys.path.append(os.path.abspath("/workspace"))

from app.utils.page_state_encoder import encode_state, decode_state, build_url

# Import visualization modules
from app.utils.pages.visu.TableVisu import TableVisu
from app.utils.pages.visu.ProfileReportVisu import ProfileReportVisu
from app.utils.pages.visu.PlotVisu import PlotVisu
from app.utils.pages.visu.RegressionVisu import RegressionVisu


# Registry of available visualizations
VISU_REGISTRY = {
    "Table": TableVisu,
    "Profile Report": ProfileReportVisu,
    "Plots": PlotVisu,
    "Regression": RegressionVisu,
}

# Page config
st.set_page_config(layout="wide", page_title="Data Viewer", page_icon="📊")


# ============================================================
# 1. Session State Initialization
# ============================================================
def init_session_state():
    """
    Initialize or restore the central session state dictionary.

    Structure:
    {
        "global": {...},         # global sidebar settings
        "visu_states": {...},    # per-visualization configs
        "subpages": {...}        # optional: other page states
    }
    """
    params = st.query_params
    initial_state = decode_state(params.get("state", "")) if "state" in params else {}

    if "full_state" not in st.session_state:
        st.session_state["full_state"] = {
            "global": {
                "db_key": initial_state.get("db_key", "production"),
                "table": initial_state.get("table"),
                "limit": initial_state.get("limit", 1000),
                "limit_mode": initial_state.get("limit_mode", "random"),
                "sort_col": initial_state.get("sort_col"),
                "sort_order": initial_state.get("sort_order", "desc"),
                "filters": initial_state.get("filters", {}),
                "visu_choice": initial_state.get("visu_choice", "Table"),
            },
            "visu_states": initial_state.get("visu_states", {}),
            "subpages": initial_state.get("subpages", {}),
        }


# ============================================================
# 2. Data Helpers
# ============================================================
@st.cache_data(ttl=300)
def get_all_table_caches():
    """Fetch cached table metadata from backend."""
    res = requests.get(f"{BASE_URL_VISU}/cache/all_table_caches")
    res.raise_for_status()
    return res.json()

@st.cache_data(ttl=600)
def fetch_data(payload):
    """Fetch table data from backend based on query payload."""
    res = requests.post(f"{BASE_URL_VISU}/data", json=payload)
    res.raise_for_status()
    return pd.DataFrame(res.json()["data"])


# ============================================================
# 3. Sidebar UI
# ============================================================
def render_sidebar():
    """
    Render sidebar controls and update global state accordingly.
    """
    global_state = st.session_state["full_state"]["global"]

    st.sidebar.title("Data Source & Query")

    # Visualization type
    visu_choice = st.sidebar.selectbox(
        "✨ Visualization",
        list(VISU_REGISTRY.keys()),
        index=list(VISU_REGISTRY.keys()).index(global_state.get("visu_choice", "Table"))
    )
    global_state["visu_choice"] = visu_choice

    # Database
    db_key = st.sidebar.selectbox(
        "Database", ["raw", "source", "production"],
        index=["raw", "source", "production"].index(global_state.get("db_key", "raw"))
    )
    global_state["db_key"] = db_key

    # Table options
    tables = list(st.session_state["all_table_caches"].get(db_key, {}).keys())
    if not tables:
        st.sidebar.error("No cached tables for this DB – run the cache task first.")
        st.stop()
        
    def safe_index(options, value, default=0):
        try:
            return options.index(value)
        except (ValueError, AttributeError):
            return default

    table = global_state.get("table") or tables[0]
    table = st.sidebar.selectbox("Table", tables, index=safe_index(tables, table))

    global_state["table"] = table

    # Row limit
    limit_default = int(global_state.get("limit", 1000))

    # clamp the value to slider bounds
    if limit_default < 100:
        limit_default = 100
    elif limit_default > 50000:
        limit_default = 50000

    row_limit = st.sidebar.slider(
        "Row Limit", min_value=100, max_value=50000, step=1000,
        value=limit_default
    )

    global_state["limit"] = row_limit


    # Limit mode
    limit_mode = st.sidebar.selectbox(
        "Limit Mode", ["top", "bottom", "random"],
        index=["top", "bottom", "random"].index(global_state.get("limit_mode", "top"))
    )
    global_state["limit_mode"] = limit_mode

    # Sort column (only for numeric cols)
    columns = st.session_state["all_table_caches"].get(db_key, {}).get(table, {}).get("columns", [])
    numeric_cols = [c["name"] for c in columns if c.get("type") == "numeric"]
    sort_col, sort_order = None, "desc"
    if limit_mode in ["top", "bottom"] and numeric_cols:
        sort_col = st.sidebar.selectbox("Sort Column", numeric_cols)
        sort_order = "asc" if limit_mode == "bottom" else "desc"
    global_state["sort_col"] = sort_col
    global_state["sort_order"] = sort_order

    # Filters
    st.sidebar.markdown("### 🔧 Filters")
    filters = {}
    if st.sidebar.checkbox("Enable Filters", value=True):
        with st.sidebar.expander("🔧 Filters", expanded=False):
            for col in columns:
                col_name, col_type = col["name"], col.get("type")
                if col_type == "numeric":
                    min_val, max_val = col.get("min"), col.get("max")
                    if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)) and min_val != max_val:
                        sel_min, sel_max = st.slider(col_name, min_value=min_val, max_value=max_val, value=(min_val, max_val))
                        filters[col_name] = {"type": "numeric", "min": sel_min, "max": sel_max}
                elif col_type == "categorical":
                    uniq_vals, uniq_cnt = col.get("unique_values", []), col.get("unique_count", 0)
                    if uniq_cnt <= 10 and uniq_vals:
                        chosen = st.multiselect(col_name, uniq_vals)
                        if chosen:
                            filters[col_name] = {"type": "categorical", "values": chosen}
                    else:
                        txt = st.text_input(f"Contains {col_name}")
                        if txt:
                            filters[col_name] = {"type": "categorical", "values": [txt]}
    global_state["filters"] = filters

    # Fetch button
    return st.sidebar.button("🚀 Fetch Data", type="primary")


# ============================================================
# 4. Main Content
# ============================================================
def render_main(fetch_btn: bool):
    """
    Render main content:
      - Fetch data if requested
      - Dispatch visualization
      - Update and show shareable link
    """
    global_state = st.session_state["full_state"]["global"]
    visu_choice = global_state["visu_choice"]


    # --- Auto-fetch if entering via shared state ---
    has_state_in_url = "state" in st.query_params
    if (fetch_btn or (has_state_in_url and "last_fetched_df" not in st.session_state)):
        payload = {
            "db_key": global_state["db_key"],
            "table": global_state["table"],
            "limit": global_state["limit"],
            "limit_mode": global_state["limit_mode"],
            "sort_col": global_state.get("sort_col"),
            "sort_order": global_state.get("sort_order"),
            "filters": global_state.get("filters", {}),
        }
        with st.spinner("🔄 Fetching data …"):
            df = fetch_data(payload)
        st.toast(f"Fetched {len(df)} rows.")
        st.session_state["last_fetched_df"] = df

    # Fetch data
    if fetch_btn:
        payload = {
            "db_key": global_state["db_key"],
            "table": global_state["table"],
            "limit": global_state["limit"],
            "limit_mode": global_state["limit_mode"],
            "sort_col": global_state.get("sort_col"),
            "sort_order": global_state.get("sort_order"),
            "filters": global_state.get("filters", {}),
        }
        with st.spinner("🔄 Fetching data …"):
            df = fetch_data(payload)
        st.toast(f"Fetched {len(df)} rows.")
        st.session_state["last_fetched_df"] = df

    # Render visualization if data available
    if "last_fetched_df" in st.session_state:
        df = st.session_state["last_fetched_df"]
        visu_cls = VISU_REGISTRY[visu_choice]

        # Pass initial config from state
        initial_config = st.session_state["full_state"]["visu_states"].get(visu_choice, {})

        visu_instance = visu_cls(
            df,
            shared_state={"db": global_state["db_key"], "table": global_state["table"]},
            initial_config=initial_config
        )
        visu_instance.render()

        # Collect updated visu state
        visu_state = visu_instance.get_state() if hasattr(visu_instance, "get_state") else {}
        st.session_state["full_state"]["visu_states"][visu_choice] = visu_state

    # Shareable link
    share_url = build_url("02_Visualization", st.session_state["full_state"])
    st.sidebar.markdown("### 🔗 Shareable Link")
    st.sidebar.code(share_url, language="text")


# ============================================================
# Page Execution (multi-page mode)
# ============================================================
# Init state
init_session_state()

# Load metadata if not available
if "all_table_caches" not in st.session_state:
    with st.spinner("🔄 Loading column metadata …"):
        st.session_state["all_table_caches"] = get_all_table_caches()

# Render UI
fetch_btn = render_sidebar()
render_main(fetch_btn)
