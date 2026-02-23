FROM python:3.10-slim
LABEL description="T4_synthetic_validation: RTM Black Holes Framework"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY T4_synthetic_validation.py .
RUN mkdir -p output
CMD ["python", "T4_synthetic_validation.py"]
