FROM python:3.10-slim
LABEL description="S3_btfr_residuals: Rhythmic Astronomy RTM Simulation"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S3_btfr_residuals.py .
RUN mkdir -p output
CMD ["python", "S3_btfr_residuals.py"]
