import os, sys, json, base64
import pandas as pd
import streamlit as st
import requests
from urllib.parse import quote, unquote
import yaml

from utils.hosts import BASE_URL_VISU




sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from app.utils.pages.visu.TableVisu import TableVisu
from app.utils.pages.visu.ProfileReportVisu import ProfileReportVisu
from app.utils.pages.visu.PlotVisu import PlotVisu
from app.utils.pages.visu.RegressionVisu import RegressionVisu


def embedded_panel():
    # Ensure all fields from Streamlit sidebar are included
    import os
    print(f"🔔 Current working directory: {os.getcwd()}", flush=True)
    session_state = {
        "db_key": db_key,
        "table": table,
        "limit": row_limit,
        "limit_mode": limit_mode,
        "sort_col": sort_col,
        "sort_order": sort_order,
        "filters": filters,
    }

    # Write session state to shared YAML file
    with open("shared/session_state.yaml", "w") as f:
        yaml.safe_dump(session_state, f)

    # Embed iframe pointing to Panel (no query params)
    panel_url = "http://localhost:5006/"
    st.components.v1.iframe(panel_url, height=1000)



VISU_REGISTRY = {
    "Table": TableVisu,
    "Profile Report": ProfileReportVisu,
    "Plots": PlotVisu,
    "Regression": RegressionVisu
#    "Panel": "Embedded Panel",
}

st.set_page_config(layout="wide", page_title="Data Viewer", page_icon="📊")

@st.cache_data(ttl=300)
def get_all_table_caches():
    res = requests.get(f"{BASE_URL_VISU}/cache/all_table_caches")
    res.raise_for_status()
    return res.json()

@st.cache_data(ttl=600)
def fetch_data(payload):
    if visu_choice == "Panel":
        # For Panel, we don't fetch data here, it will be handled in the embedded app
        return pd.DataFrame()


    res = requests.post(f"{BASE_URL_VISU}/data", json=payload)
    res.raise_for_status()
    return pd.DataFrame(res.json()["data"])

if "all_table_caches" not in st.session_state:
    with st.spinner("🔄 Loading column metadata …"):
        st.session_state["all_table_caches"] = get_all_table_caches()

st.sidebar.title("Data Source & Query")

visu_choice = st.sidebar.selectbox("✨ Visualization", list(VISU_REGISTRY.keys()), index=0)



fetch_btn   = st.sidebar.button("🚀 Fetch Data", type="primary")

db_key = st.sidebar.selectbox("Database", ["raw", "source", "production"])

tables = list(st.session_state["all_table_caches"].get(db_key, {}).keys())
if not tables:
    st.sidebar.error("No cached tables for this DB – run the cache task first.")
    st.stop()

table = st.sidebar.selectbox("Table", tables)
row_limit = st.sidebar.slider("Row Limit", 100, 50000, 1000)

limit_mode = st.sidebar.selectbox("Limit Mode", ["top", "bottom", "random"])
sort_col = None
sort_order = "desc"

columns = st.session_state["all_table_caches"].get(db_key, {}).get(table, {}).get("columns", [])
numeric_cols = [c["name"] for c in columns if c.get("type") == "numeric"]
if limit_mode in ["top", "bottom"] and numeric_cols:
    sort_col = st.sidebar.selectbox("Sort Column", numeric_cols)
    sort_order = "asc" if limit_mode == "bottom" else "desc"


st.sidebar.markdown("### 🔧 Filters")
use_filters = st.sidebar.checkbox("Enable Filters", value=True)

filters = {}
if use_filters:
    with st.sidebar.expander("🔧 Filters", expanded=False):
        for col in columns:
            col_name = col["name"]
            col_type = col.get("type")
            if col_type == "numeric":
                min_val, max_val = col.get("min"), col.get("max")
                if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)) and min_val != max_val:
                    sel_min, sel_max = st.slider(col_name, min_value=min_val, max_value=max_val, value=(min_val, max_val))
                    filters[col_name] = {"type": "numeric", "min": sel_min, "max": sel_max}
            elif col_type == "categorical":
                uniq_vals = col.get("unique_values", [])
                uniq_cnt = col.get("unique_count", 0)
                if uniq_cnt <= 10 and uniq_vals:
                    chosen = st.multiselect(col_name, uniq_vals)
                    if chosen:
                        filters[col_name] = {"type": "categorical", "values": chosen}
                else:
                    txt = st.text_input(f"Contains {col_name}")
                    if txt:
                        filters[col_name] = {"type": "categorical", "values": [txt]}

if fetch_btn:
    payload = {
        "db_key": db_key,
        "table": table,
        "limit": row_limit,
        "limit_mode": limit_mode,
        "sort_col": sort_col,
        "sort_order": sort_order,
        "filters": filters,
    }
    with st.spinner("🔄 Fetching data …"):
        df = fetch_data(payload)
    st.toast(f"Fetched {len(df)} rows.")
    st.session_state["last_fetched_df"] = df

if "last_fetched_df" in st.session_state:
    if visu_choice == "Panel":
        embedded_panel()
    else:

        df = st.session_state["last_fetched_df"]
        visu_cls = VISU_REGISTRY[visu_choice]
        visu_instance = visu_cls(df, shared_state={"db": db_key, "table": table})
        visu_instance.render()