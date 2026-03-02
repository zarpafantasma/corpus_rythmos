FROM python:3.10-slim
LABEL description="S2_threshold_matching: RTM Unified Field Framework RG Unification"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S2_threshold_matching.py .
RUN mkdir -p output
CMD ["python", "S2_threshold_matching.py"]
