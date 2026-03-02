# E2: Directional Causality in a Layered Chain
# RTM Cascade Framework - Signature S2 Validation

FROM python:3.10-slim

LABEL maintainer="RTM Corpus"
LABEL description="E2 Simulation: Directional Causality (TE/Granger)"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY E2_directional_causality.py .
COPY E2_directional_causality.ipynb .

RUN mkdir -p output

CMD ["python", "E2_directional_causality.py"]
