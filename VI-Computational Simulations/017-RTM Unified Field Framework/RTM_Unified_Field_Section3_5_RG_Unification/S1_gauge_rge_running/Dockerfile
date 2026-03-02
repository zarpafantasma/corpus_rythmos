FROM python:3.10-slim
LABEL description="S1_gauge_rge_running: RTM Unified Field Framework RG Unification"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S1_gauge_rge_running.py .
RUN mkdir -p output
CMD ["python", "S1_gauge_rge_running.py"]
