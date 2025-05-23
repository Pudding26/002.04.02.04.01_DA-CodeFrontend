import streamlit as st
import requests

ORCHESTRATOR_URL = "http://localhost:8000"

st.title("Mockup Frontend")

user_input = st.text_input("Enter input for modeling task:")

if st.button("Submit"):
    try:
        response = requests.post(
            f"{ORCHESTRATOR_URL}/run-job",
            json={"input": user_input}
        )
        result = response.json()
        st.success(f"Job result: {result['output']}")
    except Exception as e:
        st.error(f"Error: {e}")
