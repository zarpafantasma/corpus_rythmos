FROM python:3.10-slim
LABEL description="S1_calorimetric_power: RTM Experimental Signatures"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S1_calorimetric_power.py .
RUN mkdir -p output
CMD ["python", "S1_calorimetric_power.py"]
