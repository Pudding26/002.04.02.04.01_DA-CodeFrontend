FROM python:3.11-slim

WORKDIR /app
COPY . .

# Install system packages
RUN apt-get update && \
    apt-get install -y hdf5-tools && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


CMD ["env", "STREAMLIT_EMAIL=no", "streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
