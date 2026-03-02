# E1: Four-Layer Cascade with Non-Decreasing Coherence
# RTM Cascade Framework - Signature S1 Validation

FROM python:3.10-slim

LABEL maintainer="RTM Corpus"
LABEL description="E1 Simulation: Monotone Coherence Cascade"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy simulation files
COPY E1_monotone_coherence.py .
COPY E1_monotone_coherence.ipynb .

# Create output directory
RUN mkdir -p output

# Run simulation
CMD ["python", "E1_monotone_coherence.py"]
