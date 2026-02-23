FROM python:3.10-slim
LABEL description="S1_static_thrust: Aetherion Chapter II Simulation"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S1_static_thrust.py .
RUN mkdir -p output
CMD ["python", "S1_static_thrust.py"]
