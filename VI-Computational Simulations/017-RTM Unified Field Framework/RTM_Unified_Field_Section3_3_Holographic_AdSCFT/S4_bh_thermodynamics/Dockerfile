FROM python:3.10-slim
LABEL description="S4_bh_thermodynamics: RTM Holographic/AdS-CFT"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S4_bh_thermodynamics.py .
RUN mkdir -p output
CMD ["python", "S4_bh_thermodynamics.py"]
