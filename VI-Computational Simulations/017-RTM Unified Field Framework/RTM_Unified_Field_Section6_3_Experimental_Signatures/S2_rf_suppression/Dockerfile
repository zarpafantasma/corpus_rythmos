FROM python:3.10-slim
LABEL description="S2_rf_suppression: RTM Experimental Signatures"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S2_rf_suppression.py .
RUN mkdir -p output
CMD ["python", "S2_rf_suppression.py"]
