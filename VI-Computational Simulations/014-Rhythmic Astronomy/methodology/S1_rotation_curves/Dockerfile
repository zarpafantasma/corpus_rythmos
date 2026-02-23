FROM python:3.10-slim
LABEL description="S1_rotation_curves: Rhythmic Astronomy RTM Simulation"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S1_rotation_curves.py .
RUN mkdir -p output
CMD ["python", "S1_rotation_curves.py"]
