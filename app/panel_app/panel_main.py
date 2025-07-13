# main_app.py (stripped-down starter module)
import os
import json
import hashlib
import yaml
import pandas as pd
import panel as pn
import requests

from plot_control import build_controls

pn.extension('tabulator', sizing_mode='stretch_width')

BACKEND_HOST = os.getenv('BACKEND_ORCH_BASE_URL', 's003-c9501_lele-da-app')
BACKEND_PORT = os.getenv('BACKEND_ORCH_BASE_PORT', '59501')
BASE_URL = f'http://{BACKEND_HOST}:{BACKEND_PORT}/visu'
API_URL = f'{BASE_URL}/data'
SESSION_STATE_FILE = '/workspace/shared/session_state.yaml'

_current_df = pd.DataFrame()
_current_state_hash = None

plots_area = pn.Column()

controls = build_controls(plots_area)

# -----------------------------------------------------------
# 📡 Data loader + refresher
# -----------------------------------------------------------

def _fetch_data():
    global _current_state_hash, _current_df
    try:
        with open(SESSION_STATE_FILE, 'r') as f:
            state = yaml.safe_load(f) or {}
        state_hash = hashlib.md5(json.dumps(state, sort_keys=True).encode()).hexdigest()
        if state_hash != _current_state_hash:
            _current_state_hash = state_hash
            resp = requests.post(API_URL, json=state, timeout=15)
            resp.raise_for_status()
            _current_df = pd.DataFrame(resp.json().get('data', []))
            controls.update_column_options(_current_df)
    except Exception as exc:
        plots_area.objects = [pn.pane.Markdown(f"❌ Failed to fetch data: {exc}")]

pn.state.add_periodic_callback(_fetch_data, period=2000)

layout = pn.Column(controls, plots_area)
_main_layout = pn.Column("🔄 Waiting for session state...", layout)
_main_layout.servable()
