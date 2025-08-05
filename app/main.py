import streamlit as st
import requests
from dotenv import load_dotenv
import os
import logging


import sys, os
sys.path.append(os.path.abspath("/workspace"))
from app.utils.hosts import ORCHESTRATOR_URL


# Set global default log level
logging.basicConfig(
    level=logging.INFO,  # default: show INFO and above
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Silence noisy third-party modules

# Force all watchdog loggers to WARNING
for name in logging.root.manager.loggerDict:
    if name.startswith("watchdog"):
        logging.getLogger(name).setLevel(logging.WARNING)

logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

# If you want your plot logs at DEBUG, configure just that namespace
logging.getLogger("plotbase").setLevel(logging.DEBUG)
logging.getLogger("plotvisu").setLevel(logging.DEBUG)

import logging

class NullHandler(logging.Handler):
    def emit(self, record):
        pass

# Attach a null handler to all watchdog loggers
for name in list(logging.root.manager.loggerDict.keys()):
    if name.startswith("watchdog"):
        logging.getLogger(name).handlers = [NullHandler()]
        logging.getLogger(name).propagate = False


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

