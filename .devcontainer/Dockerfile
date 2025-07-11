FROM python:3.10-slim

# Install Node.js (for frontend if needed)
RUN apt-get update && \
    apt-get install -y curl git && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean

# Backend dependencies
RUN apt-get update && \
    apt-get install -y hdf5-tools && \
    rm -rf /var/lib/apt/lists/*

# Install Java and Graphviz properly
RUN apt-get update && \
    apt-get install -y default-jre graphviz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*



WORKDIR /workspace
