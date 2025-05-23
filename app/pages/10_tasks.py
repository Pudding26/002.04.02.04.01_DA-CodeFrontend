import streamlit as st
import httpx


st.title("Available Tasks")

# Load tasks only once per session
if "task_data" not in st.session_state:
    response = httpx.get("http://localhost:8000/tasks/")
    st.session_state.task_data = response.json()["possible_tasks"]

task_data = st.session_state.task_data
col1, col2 = st.columns([1, 3])

with col1:
    if st.button("🔁 Reload Instructions", key="reload_btn"):
        try:
            resp = httpx.post("http://localhost:8000/tasks/reload", timeout=5)
            st.session_state["reload_success"] = resp.json().get("message", "Reloaded")
            # Clear task list so it reloads on next render
            if "task_data" in st.session_state:
                del st.session_state["task_data"]
            st.rerun()
        except Exception as e:
            st.session_state["reload_success"] = f"❌ {e}"

# Show the message next to the button (if it exists)
with col2:
    if "reload_success" in st.session_state:
        st.success(st.session_state["reload_success"])

# UI: Task group and task selection
selected_group = st.selectbox("Select Task Group", [g["group"] for g in task_data])

selected_task = None
for group in task_data:
    if group["group"] == selected_group:
        selected_task = st.selectbox("Select Task", group["tasks"])

# Optional reload button (clears and re-fetches)
if st.button("🔄 Reload task list"):
    del st.session_state["task_data"]
    st.rerun()

# Display selected task
if st.button("Show Task Info"):
    st.write(f"Group: {selected_group}")
    st.write(f"Task: {selected_task}")

if st.button("🚀 Run Task"):
    resp = httpx.post("http://localhost:8000/tasks/start", json={"task_name": selected_task})
    st.success(resp.json())


if st.button("🛑 Stop Task"):
    resp = httpx.post("http://localhost:8000/tasks/stop", json={"task_name": selected_task})
    st.warning(resp.json())


response = httpx.get("http://localhost:8000/tasks/status")
tasks = response.json()

for task in tasks:
    @st.fragment
    def render_task(task=task):
        with st.expander(task["task_name"]):
            st.write(f"Status: {task['status']}")
            st.write(f"Message: {task['message']}")
            st.progress(task["progress"])

            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"⏹ Stop {task['task_name']}", key=f"stop_{task['task_name']}"):
                    httpx.post("http://localhost:8000/tasks/stop", json={"task_name": task["task_name"]})
            with col2:
                toggle_label = "⏸ Pause" if task["status"] != "Paused" else "▶️ Resume"
                if st.button(f"{toggle_label} {task['task_name']}", key=f"pause_{task['task_name']}"):
                    action = "pause" if task["status"] != "Paused" else "resume"
                    httpx.post("http://localhost:8000/tasks/control", json={
                        "task_name": task["task_name"],
                        "action": action
                    })

            st.rerun()
    render_task(task)



