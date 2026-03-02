# RTM Simulation: Flat Small-World Network (MFPT)
# Dockerfile for reproducible execution

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy simulation files
COPY flat_small_world.py .
COPY flat_small_world.ipynb .

# Create output directory
RUN mkdir -p output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

# Default command: run the simulation
CMD ["python", "flat_small_world.py"]

# =============================================================================
# BUILD AND RUN INSTRUCTIONS
# =============================================================================
# 
# Build the image:
#   docker build -t rtm-flat-small-world .
#
# Run the simulation:
#   docker run --rm -v $(pwd)/output:/app/output rtm-flat-small-world
#
# Run interactively:
#   docker run -it --rm rtm-flat-small-world bash
#
# Run Jupyter notebook:
#   docker run -p 8888:8888 -v $(pwd)/output:/app/output rtm-flat-small-world \
#       jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
#
# =============================================================================
