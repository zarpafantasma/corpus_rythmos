FROM python:3.10-slim
LABEL description="S1_1D_slab_simulation: Aetherion Chapter I Simulation"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S1_1D_slab_simulation.py .
RUN mkdir -p output
CMD ["python", "S1_1D_slab_simulation.py"]
