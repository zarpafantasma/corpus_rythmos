FROM python:3.10-slim
LABEL description="S3_rg_flow: RTM Quantum Corrections"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S3_rg_flow.py .
RUN mkdir -p output
CMD ["python", "S3_rg_flow.py"]
