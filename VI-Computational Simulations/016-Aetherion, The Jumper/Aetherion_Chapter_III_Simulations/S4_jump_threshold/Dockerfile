FROM python:3.10-slim
LABEL description="S4_jump_threshold: Aetherion Chapter III Simulation"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S4_jump_threshold.py .
RUN mkdir -p output
CMD ["python", "S4_jump_threshold.py"]
