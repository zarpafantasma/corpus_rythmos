FROM python:3.10-slim
LABEL description="S2_2D_simulation: Aetherion Chapter I Simulation"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S2_2D_simulation.py .
RUN mkdir -p output
CMD ["python", "S2_2D_simulation.py"]
