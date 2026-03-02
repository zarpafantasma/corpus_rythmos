FROM python:3.10-slim
LABEL description="S3_photon_delay: RTM Experimental Signatures"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S3_photon_delay.py .
RUN mkdir -p output
CMD ["python", "S3_photon_delay.py"]
