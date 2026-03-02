# RTM Simulation: Vascular Tree (Fractal Lattice)
# Dockerfile for reproducible execution

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vascular_tree.py .
COPY vascular_tree.ipynb .

RUN mkdir -p output

ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

CMD ["python", "vascular_tree.py"]

# =============================================================================
# BUILD AND RUN
# =============================================================================
# 
# docker build -t rtm-vascular-tree .
# docker run --rm -v $(pwd)/output:/app/output rtm-vascular-tree
#
# =============================================================================
