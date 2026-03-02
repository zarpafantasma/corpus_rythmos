# E3: Ratchet/Hysteresis Under Coupling Sweeps
# RTM Cascade Framework - Signature S3 Validation

FROM python:3.10-slim

LABEL maintainer="RTM Corpus"
LABEL description="E3 Simulation: Hysteresis in Coupling Sweeps"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY E3_hysteresis.py .
COPY E3_hysteresis.ipynb .

RUN mkdir -p output

CMD ["python", "E3_hysteresis.py"]
