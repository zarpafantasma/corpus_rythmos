# RTM Simulation: Sierpiński Fractal Network
# Dockerfile for reproducible execution

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sierpinski_fractal.py .
COPY sierpinski_fractal.ipynb .

RUN mkdir -p output

ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

CMD ["python", "sierpinski_fractal.py"]

# =============================================================================
# BUILD AND RUN
# =============================================================================
# 
# docker build -t rtm-sierpinski .
# docker run --rm -v $(pwd)/output:/app/output rtm-sierpinski
#
# =============================================================================
