FROM python:3.10-slim
LABEL description="S2_holographic_rg_flow: RTM Holographic/AdS-CFT"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S2_holographic_rg_flow.py .
RUN mkdir -p output
CMD ["python", "S2_holographic_rg_flow.py"]
