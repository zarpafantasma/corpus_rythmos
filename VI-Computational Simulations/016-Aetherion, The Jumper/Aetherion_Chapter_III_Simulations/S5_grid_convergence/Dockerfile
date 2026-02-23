FROM python:3.10-slim
LABEL description="S5_grid_convergence: Aetherion Chapter III Simulation"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S5_grid_convergence.py .
RUN mkdir -p output
CMD ["python", "S5_grid_convergence.py"]
