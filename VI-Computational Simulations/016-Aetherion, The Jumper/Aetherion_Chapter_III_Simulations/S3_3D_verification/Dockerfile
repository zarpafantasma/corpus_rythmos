FROM python:3.10-slim
LABEL description="S3_3D_verification: Aetherion Chapter III Simulation"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S3_3D_verification.py .
RUN mkdir -p output
CMD ["python", "S3_3D_verification.py"]
