FROM python:3.10-slim
LABEL description="S5_inertial_mitigation: Aetherion Chapter II Simulation"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S5_inertial_mitigation.py .
RUN mkdir -p output
CMD ["python", "S5_inertial_mitigation.py"]
