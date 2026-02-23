FROM python:3.10-slim
LABEL description="S2_quantum_bands: RTM Quantum Corrections"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S2_quantum_bands.py .
RUN mkdir -p output
CMD ["python", "S2_quantum_bands.py"]
