FROM python:3.10-slim
LABEL description="T4: Parameter Space Explorer"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY T4_parameter_space.py .
RUN mkdir -p output
CMD ["python", "T4_parameter_space.py"]
