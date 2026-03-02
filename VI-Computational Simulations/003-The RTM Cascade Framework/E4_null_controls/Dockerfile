# E4: Null Controls (Flat Slopes and Symmetric Causality)
# RTM Cascade Framework - Negative Control Validation

FROM python:3.10-slim

LABEL maintainer="RTM Corpus"
LABEL description="E4 Simulation: Null Controls (No False Positives)"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY E4_null_controls.py .
COPY E4_null_controls.ipynb .

RUN mkdir -p output

CMD ["python", "E4_null_controls.py"]
