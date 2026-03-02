FROM python:3.10-slim
LABEL description="S3_boundary_correlators: RTM Holographic/AdS-CFT"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S3_boundary_correlators.py .
RUN mkdir -p output
CMD ["python", "S3_boundary_correlators.py"]
