FROM python:3.10-slim
LABEL description="S2_qec_scaling: RTM-Aware Quantum Computing"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S2_qec_scaling.py .
RUN mkdir -p output
CMD ["python", "S2_qec_scaling.py"]
