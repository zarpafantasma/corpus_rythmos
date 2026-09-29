# ==============================================================================
# RTM Simulation G: Holographic Decay Network P(r) ∝ r⁻³
# Docker Container for Reproducible Execution
# ==============================================================================
# Build:  docker build -t rtm-holographic-decay .
# Run:    docker run --rm -v $(pwd)/output:/app/output rtm-holographic-decay
# Jupyter: docker run --rm -p 8888:8888 rtm-holographic-decay jupyter
# ==============================================================================

FROM python:3.12-slim

LABEL maintainer="RTM Framework – Computational Validation Suite"
LABEL description="Holographic Decay Network Simulation (α → 3.0)"
LABEL license="CC BY 4.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy simulation files
COPY holographic_decay_simulation.py .
COPY holographic_decay_notebook.ipynb .

# Create output directories
RUN mkdir -p /app/output/figures

# Default: run the simulation
ENTRYPOINT ["python3"]
CMD ["holographic_decay_simulation.py", "--output-dir", "/app/output"]

# Alternative: Jupyter notebook
# Override CMD with: docker run -p 8888:8888 <image> jupyter notebook \
#   --ip=0.0.0.0 --port=8888 --no-browser --allow-root
