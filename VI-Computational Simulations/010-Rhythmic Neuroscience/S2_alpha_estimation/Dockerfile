FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S2_alpha*.py .
CMD ["python", "S2_alpha.py"]
