FROM python:3.10-slim
LABEL description="S3_unification_fit: RTM Unified Field Framework RG Unification"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S3_unification_fit.py .
RUN mkdir -p output
CMD ["python", "S3_unification_fit.py"]
