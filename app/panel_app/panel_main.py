import os
import panel as pn
import pandas as pd
import hvplot.pandas
import requests
import yaml
import hashlib
import json

pn.extension()

BACKEND_HOST = os.getenv("BACKEND_ORCH_BASE_URL", "s003-c9501_lele-da-app")
BACKEND_PORT = os.getenv("BACKEND_ORCH_BASE_PORT", "59501")
BASE_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}/visu"
API_URL = f"{BASE_URL}/data"

SESSION_STATE_FILE = "/workspace/shared/session_state.yaml"

main_layout = pn.Column("🔄 Waiting for session state...")

current_state_hash = None
current_df = pd.DataFrame()

# Widgets
plot_type = pn.widgets.Select(name='Plot Type', options=['Barplot', 'Lineplot', 'Scatterplot', 'Histogram', 'Heatmap'])
x_col = pn.widgets.Select(name='X Column', options=[])
y_col = pn.widgets.Select(name='Y Column', options=[])
hue_col = pn.widgets.Select(name='Hue Column (optional)', options=[''])
row_facet = pn.widgets.MultiSelect(name='Row Facet(s)', options=[])
fast_mode = pn.widgets.Checkbox(name='⚡ Fast mode (sample 100 rows)')

def compute_state_hash(state_dict):
    state_json = json.dumps(state_dict, sort_keys=True)
    return hashlib.md5(state_json.encode()).hexdigest()

def read_session_state():
    try:
        with open(SESSION_STATE_FILE, "r") as f:
            state = yaml.safe_load(f)
            return state if state else {}
    except Exception as e:
        print(f"⚠️ Could not read session state: {e}", flush=True)
        return {}

def update_widgets(df):
    options = df.columns.tolist()
    x_col.options = options
    y_col.options = options
    row_facet.options = options
    hue_col.options = [''] + options
    if options:
        x_col.value = options[0]
        y_col.value = options[1] if len(options) > 1 else options[0]

def fetch_and_update():
    global current_state_hash, current_df

    state = read_session_state()
    if not state:
        return

    state_hash = compute_state_hash(state)
    if state_hash == current_state_hash:
        return

    current_state_hash = state_hash
    print(f"🔔 New session state detected: {state}", flush=True)

    payload = {
        "db_key": state.get("db_key", "raw"),
        "table": state.get("table", "my_table"),
        "limit": int(state.get("limit", 1000)),
        "limit_mode": state.get("limit_mode", "top"),
        "sort_col": state.get("sort_col"),
        "sort_order": state.get("sort_order", "desc"),
        "filters": state.get("filters", {}),
    }

    try:
        resp = requests.post(API_URL, json=payload)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        df = pd.DataFrame(data)
    except Exception as e:
        main_layout[:] = [pn.pane.Markdown(f"❌ Failed to fetch data: {e}")]
        return

    if df.empty:
        main_layout[:] = [pn.pane.Markdown("⚠️ No data returned for this query.")]
        return

    current_df = df
    update_widgets(df)

def render_plot(**kwargs):
    if current_df.empty:
        return pn.pane.Markdown("⚠️ No data to plot yet.")

    df = current_df.copy()

    if kwargs.get('fast_mode'):
        df = df.sample(min(len(df), 100))

    plot_kind = kwargs['plot_type']
    x = kwargs['x_col']
    y = kwargs['y_col']
    hue = kwargs['hue_col'] if kwargs['hue_col'] else None
    facets = kwargs['row_facet']

    hv_opts = dict(height=500, width=800, shared_axes=False, legend='top')

    if plot_kind == 'Barplot':
        plot = df.hvplot.bar(x=x, y=y, by=facets, groupby=hue, **hv_opts)
    elif plot_kind == 'Lineplot':
        plot = df.hvplot.line(x=x, y=y, by=facets, groupby=hue, **hv_opts)
    elif plot_kind == 'Scatterplot':
        plot = df.hvplot.scatter(x=x, y=y, by=facets, groupby=hue, **hv_opts)
    elif plot_kind == 'Histogram':
        col = x if pd.api.types.is_numeric_dtype(df[x]) else y
        plot = df.hvplot.hist(col, by=facets, groupby=hue, **hv_opts)
    elif plot_kind == 'Heatmap':
        corr = df.select_dtypes(include='number').corr()
        plot = corr.hvplot.heatmap(cmap='viridis', height=500, width=500)
    else:
        plot = pn.pane.Markdown(f"❌ Unknown plot type: {plot_kind}")

    return plot

plot_panel = pn.bind(
    render_plot,
    plot_type=plot_type,
    x_col=x_col,
    y_col=y_col,
    hue_col=hue_col,
    row_facet=row_facet,
    fast_mode=fast_mode
)

layout = pn.Column(
    pn.Row(plot_type, fast_mode),
    pn.Row(x_col, y_col, hue_col),
    pn.Row(row_facet),
    plot_panel
)

main_layout[:] = [layout]

pn.state.add_periodic_callback(fetch_and_update, period=2000)
main_layout.servable()
