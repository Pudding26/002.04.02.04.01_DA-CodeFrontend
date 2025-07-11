import os, sys, json, base64
import pandas as pd
import streamlit as st
import requests
from urllib.parse import quote, unquote

# -------------------------------
# Locked Config (do not touch)
# -------------------------------
BACKEND_HOST = os.getenv("BACKEND_ORCH_BASE_URL", "s003-c9501_lele-da-app")
BACKEND_HOST = "s003-c9501_lele-da-app"
BACKEND_PORT = os.getenv("BACKEND_ORCH_BASE_PORT", "59501")
BASE_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}/visu"
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:59502")

# -------------------------------
# Visualization Registry
# -------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from app.utils.pages.visu.TableVisu import TableVisu
from app.utils.pages.visu.HistogramVisu import HistogramVisu
from app.utils.pages.visu.SankeyVisu import SankeyVisu
from app.utils.pages.visu.ProfileReportVisu import ProfileReportVisu

VISU_REGISTRY = {
    "Table": TableVisu,
    "Histogram": HistogramVisu,
    "Sankey": SankeyVisu,
    "Profile Report": ProfileReportVisu,
}

st.set_page_config(
    layout="wide",
    page_title="Data Viewer",
    page_icon="📊"
)

# -------------------------------
# Utility: Encode/Decode State
# -------------------------------
def encode_state(state: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(state).encode()).decode()

def decode_state(encoded: str) -> dict:
    try:
        return json.loads(base64.urlsafe_b64decode(encoded.encode()).decode())
    except Exception as e:
        st.warning("⚠️ Failed to decode state from URL.")
        return {}

def build_url(path: str, state: dict, base: str = PUBLIC_URL) -> str:
    query = f"state={quote(encode_state(state))}"
    return f"{base}/{path}?{query}"




# -------------------------------
# Parse state from ?state=
# -------------------------------
params = st.query_params
state = decode_state(params.get("state", "")) if "state" in params else {}

# Shared settings
db_key = state.get("db", "raw")
table = state.get("table")
chart_type = state.get("chart", "Table")
row_limit = int(state.get("limit", 1000))
initial_visible_cols = state.get("cols", [])

initial_config_dict = state


# -------------------------------
# Sidebar: Shareable Link Display
# -------------------------------
with st.sidebar:
    st.markdown("### 🔗 Shareable Link")
    st.code(st.session_state.get("share_url", "No link yet"), language="text")

# -------------------------------
# Sidebar: UI Controls
# -------------------------------
st.sidebar.title("Data Source")

db_key = st.sidebar.selectbox(
    "Select Database",
    ["raw", "source", "production"],
    index=["raw", "source", "production"].index(db_key)
)

@st.cache_data(ttl=3600)
def get_tables(db_key):
    response = requests.get(f"{BASE_URL}/tables", params={"db_key": db_key})
    response.raise_for_status()
    return response.json().get("tables", [])

tables = get_tables(db_key)
if table not in tables:
    table = tables[0]

table = st.sidebar.selectbox("Select Table", tables, index=tables.index(table))
row_limit = st.sidebar.slider("Row Limit", min_value=100, max_value=50000, step=1000, value=row_limit)
chart_type = st.sidebar.selectbox("Select Chart Type", list(VISU_REGISTRY.keys()), index=list(VISU_REGISTRY.keys()).index(chart_type))

# -------------------------------
# Data Fetching
# -------------------------------
@st.cache_data(ttl=600)
def fetch_data(db_key, table, limit):
    response = requests.post(
        f"{BASE_URL}/data",
        json={"db_key": db_key, "table": table, "limit": limit}
    )
    response.raise_for_status()
    return pd.DataFrame(response.json()["data"])

# -------------------------------
# Main: Visualization Dispatch
# -------------------------------
if db_key and table:
    df = fetch_data(db_key, table, row_limit)
    st.success(f"Loaded {len(df)} rows from `{table}`")

    shared_state = {
        "db": db_key,
        "table": table,
        "limit": row_limit,
        "chart": chart_type,
    }

    visu_class = VISU_REGISTRY.get(chart_type)
    if visu_class:
        visu_instance = visu_class(df, shared_state, initial_config=initial_config_dict)


        # Optional: inject visible cols (if set in URL)
        if hasattr(visu_instance, "set_initial_cols") and initial_visible_cols:
            visu_instance.set_initial_cols(initial_visible_cols)

        visu_instance.render()

        # Collect additional state
        extra_state = visu_instance.get_state() if hasattr(visu_instance, "get_state") else {}

        # Merge & encode full URL
        full_state = {**shared_state, **extra_state}
        share_url = build_url("Visualization", full_state)
        st.session_state["share_url"] = share_url
    else:
        st.error(f"No visualization found for chart type: {chart_type}")
