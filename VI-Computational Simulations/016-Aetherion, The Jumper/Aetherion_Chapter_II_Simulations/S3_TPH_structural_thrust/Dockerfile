FROM python:3.10-slim
LABEL description="S3_TPH_structural_thrust: Aetherion Chapter II Simulation"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S3_TPH_structural_thrust.py .
RUN mkdir -p output
CMD ["python", "S3_TPH_structural_thrust.py"]
