import streamlit as st
import requests

# Config
ORCHESTRATOR_URL = "http://localhost:8000"

st.title("🛠️ Task Handler")

@st.cache_data
def fetch_tasks():
    try:
        response = requests.get(f"{ORCHESTRATOR_URL}/tasks/")
        response.raise_for_status()
        return response.json()["possible_tasks"]
    except Exception as e:
        st.error(f"Failed to fetch tasks: {e}")
        return []

try:
    res = requests.get(f"{ORCHESTRATOR_URL}/tasks")
    res.raise_for_status()
    tasks = res.json()["possible_tasks"]
except Exception as e:
    st.error(f"❌ Failed to fetch tasks: {e}")


# Load tasks from backend
tasks = fetch_tasks()

if tasks:
    selected_task = st.selectbox("Select a task to run:", tasks)

    col1, col2 = st.columns(2)

    if col1.button("Start Task"):
        try:
            res = requests.post(f"{ORCHESTRATOR_URL}/tasks/start", json={"task_name": selected_task})
            st.success(f"Task '{selected_task}' started: {res.json()}")
        except Exception as e:
            st.error(f"Could not start task: {e}")

    if col2.button("Stop Task"):
        try:
            res = requests.post(f"{ORCHESTRATOR_URL}/tasks/stop", json={"task_name": selected_task})
            st.warning(f"Task '{selected_task}' stopped: {res.json()}")
        except Exception as e:
            st.error(f"Could not stop task: {e}")
else:
    st.info("No tasks available.")
