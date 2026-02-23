FROM python:3.10-slim
LABEL description="S4_collapse_test: RTM-Aware Quantum Computing"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S4_collapse_test.py .
RUN mkdir -p output
CMD ["python", "S4_collapse_test.py"]
