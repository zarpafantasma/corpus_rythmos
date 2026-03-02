FROM python:3.10-slim
LABEL description="S1_coleman_weinberg: RTM Quantum Corrections"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S1_coleman_weinberg.py .
RUN mkdir -p output
CMD ["python", "S1_coleman_weinberg.py"]
