FROM python:3.10-slim
LABEL description="S4_two_loop: RTM Quantum Corrections"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S4_two_loop.py .
RUN mkdir -p output
CMD ["python", "S4_two_loop.py"]
