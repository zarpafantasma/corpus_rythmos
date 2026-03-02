FROM python:3.10-slim
LABEL description="S2_OMV_vibration_thrust: Aetherion Chapter II Simulation"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S2_OMV_vibration_thrust.py .
RUN mkdir -p output
CMD ["python", "S2_OMV_vibration_thrust.py"]
