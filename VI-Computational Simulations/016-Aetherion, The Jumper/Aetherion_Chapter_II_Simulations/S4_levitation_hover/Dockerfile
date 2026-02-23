FROM python:3.10-slim
LABEL description="S4_levitation_hover: Aetherion Chapter II Simulation"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S4_levitation_hover.py .
RUN mkdir -p output
CMD ["python", "S4_levitation_hover.py"]
