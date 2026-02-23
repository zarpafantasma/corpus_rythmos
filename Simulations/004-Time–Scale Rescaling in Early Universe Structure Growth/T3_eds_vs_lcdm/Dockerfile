FROM python:3.10-slim
LABEL description="T3: EdS vs ΛCDM Comparison"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY T3_eds_vs_lcdm.py .
RUN mkdir -p output
CMD ["python", "T3_eds_vs_lcdm.py"]
