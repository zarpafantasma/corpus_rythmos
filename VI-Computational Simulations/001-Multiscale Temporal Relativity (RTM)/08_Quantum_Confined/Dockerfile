FROM python:3.12-slim

LABEL maintainer="RTM Framework"
LABEL description="RTM Simulation H: Quantum-Confined Regime (α ≈ 3.5)"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY quantum_confined_simulation.py .
COPY quantum_confined_notebook.ipynb .

RUN mkdir -p /app/output

CMD ["python", "quantum_confined_simulation.py", "--output-dir", "/app/output", "--workers", "4"]

# Usage:
#   docker build -t rtm-quantum-confined .
#   docker run --rm -v $(pwd)/output:/app/output rtm-quantum-confined
#
# Jupyter:
#   docker run --rm -p 8888:8888 rtm-quantum-confined \
#     jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
