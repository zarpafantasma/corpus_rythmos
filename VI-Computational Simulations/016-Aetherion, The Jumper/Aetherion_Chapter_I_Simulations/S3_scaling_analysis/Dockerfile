FROM python:3.10-slim
LABEL description="S3_scaling_analysis: Aetherion Chapter I Simulation"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S3_scaling_analysis.py .
RUN mkdir -p output
CMD ["python", "S3_scaling_analysis.py"]
