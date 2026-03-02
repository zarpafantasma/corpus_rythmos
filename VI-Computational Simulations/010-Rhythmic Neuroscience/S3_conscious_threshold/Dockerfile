FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S3_conscious*.py .
CMD ["python", "S3_conscious.py"]
