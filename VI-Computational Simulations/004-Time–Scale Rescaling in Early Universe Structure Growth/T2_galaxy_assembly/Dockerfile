FROM python:3.10-slim
LABEL description="T2: Galaxy Assembly Calculator"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY T2_galaxy_assembly.py .
RUN mkdir -p output
CMD ["python", "T2_galaxy_assembly.py"]
