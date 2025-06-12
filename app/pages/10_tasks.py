import os
import streamlit as st
import httpx
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()
BACKEND_HOST = os.getenv("BACKEND_ORCH_BASE_URL")
BACKEND_HOST = "localhost"
BACKEND_PORT = os.getenv("BACKEND_ORCH_BASE_PORT", "8000")
DEBUG_MODE = os.getenv("DEBUG_MODE")


BASE_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

st.title("Available Tasks")

# --- Utility Functions ---
def get_tasks():
    return httpx.get(f"{BASE_URL}/tasks/").json().get("possible_tasks", [])

def reload_instructions():
    try:
        resp = httpx.post(f"{BASE_URL}/tasks/reload", timeout=5)
        st.session_state["reload_success"] = resp.json().get("message", "Reloaded")
        st.session_state.pop("task_data", None)
        st.rerun()
    except Exception as e:
        st.session_state["reload_success"] = f"❌ {e}"

def run_task(task_name):
    return httpx.post(f"{BASE_URL}/tasks/start", json={"task_name": task_name})

def stop_task(task_name):
    return httpx.post(f"{BASE_URL}/tasks/stop", json={"task_name": task_name})

def control_task(task_name, action):
    return httpx.post(f"{BASE_URL}/tasks/control", json={"task_name": task_name, "action": action})

# --- Load Tasks Once ---
if "task_data" not in st.session_state:
    st.session_state.task_data = get_tasks()

task_data = st.session_state.task_data

# --- Reload Instructions UI ---
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🔁 Reload Instructions", key="reload_btn"):
        reload_instructions()

with col2:
    if "reload_success" in st.session_state:
        st.success(st.session_state["reload_success"])

# --- Task Selection ---
selected_group = st.selectbox("Select Task Group", [g["group"] for g in task_data])

selected_task = next(
    (st.selectbox("Select Task", group["tasks"]) for group in task_data if group["group"] == selected_group),
    None
)

# --- Task Controls ---
if st.button("🔄 Reload task list"):
    st.session_state.pop("task_data", None)
    st.rerun()

if st.button("Show Task Info"):
    st.write(f"Group: {selected_group}")
    st.write(f"Task: {selected_task}")

if st.button("🚀 Run Task"):
    resp = run_task(selected_task)
    st.success(resp.json())

if st.button("🛑 Stop Task"):
    resp = stop_task(selected_task)
    st.warning(resp.json())


# --- Fetch all task statuses once to extract task names ---
initial_response = httpx.get(f"{BASE_URL}/tasks/status")
initial_tasks = initial_response.json()
task_names = [task["task_name"] for task in initial_tasks]

# --- Fragment for each task ---
for task_name in task_names:
    @st.fragment(run_every="1s")  # reasonable polling interval
    def render_task(taskname=task_name):  # default arg avoids closure bugs
        # Fetch only the current task's status
        response = httpx.get(f"{BASE_URL}/tasks/status")
        tasks = response.json()
        task = next((t for t in tasks if t["task_name"] == taskname), None)

        if not task:
            st.warning(f"Task {taskname} not found.")
            return

        with st.expander(task["task_name"], expanded=True):
            st.write(f"**Status:** {task['status']}")
            st.write(f"**Message:** {task['message']}")
            st.progress(task["progress"])

            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"⏹ Stop", key=f"stop_{task['task_name']}"):
                    stop_task(task["task_name"])
            with col2:
                toggle_label = "⏸ Pause" if task["status"] != "Paused" else "▶️ Resume"
                if st.button(f"{toggle_label}", key=f"pause_{task['task_name']}"):
                    action = "pause" if task["status"] != "Paused" else "resume"
                    control_task(task["task_name"], action)

    # 👇 Render each fragment with the specific task name
    render_task(task_name)


