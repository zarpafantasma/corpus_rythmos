FROM python:3.10-slim
LABEL description="S3_mesh_convergence: RTM Unified Field Framework Section 4"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S3_mesh_convergence.py .
RUN mkdir -p output
CMD ["python", "S3_mesh_convergence.py"]
