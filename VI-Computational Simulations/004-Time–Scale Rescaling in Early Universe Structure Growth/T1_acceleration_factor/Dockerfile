FROM python:3.10-slim
LABEL description="T1: RTM Acceleration Factor Calculator"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY T1_acceleration_factor.py .
RUN mkdir -p output
CMD ["python", "T1_acceleration_factor.py"]
