FROM python:3.10-slim
LABEL description="S3_runtime_scaling: RTM-Aware Quantum Computing"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S3_runtime_scaling.py .
RUN mkdir -p output
CMD ["python", "S3_runtime_scaling.py"]
