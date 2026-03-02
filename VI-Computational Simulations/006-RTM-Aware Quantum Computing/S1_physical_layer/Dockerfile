FROM python:3.10-slim
LABEL description="S1_physical_layer: RTM-Aware Quantum Computing"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S1_physical_layer.py .
RUN mkdir -p output
CMD ["python", "S1_physical_layer.py"]
