FROM python:3.10-slim
LABEL description="S5_alpha_fusion: RTM-Aware Quantum Computing"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S5_alpha_fusion.py .
RUN mkdir -p output
CMD ["python", "S5_alpha_fusion.py"]
