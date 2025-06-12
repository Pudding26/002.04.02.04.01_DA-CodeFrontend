import streamlit as st
import requests
from dotenv import load_dotenv
import os


load_dotenv()

ORCHESTRATOR_URL = "http://" + os.getenv("BACKEND_ORCH_BASE_URL") + ":" + os.getenv("BACKEND_ORCH_BASE_PORT")


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
