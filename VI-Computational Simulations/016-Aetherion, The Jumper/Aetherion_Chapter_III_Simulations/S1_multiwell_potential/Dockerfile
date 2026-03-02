FROM python:3.10-slim
LABEL description="S1_multiwell_potential: Aetherion Chapter III Simulation"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S1_multiwell_potential.py .
RUN mkdir -p output
CMD ["python", "S1_multiwell_potential.py"]
