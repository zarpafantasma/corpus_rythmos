FROM python:3.10-slim
LABEL description="T2_slope_estimator: RTM Black Holes Framework"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY T2_slope_estimator.py .
RUN mkdir -p output
CMD ["python", "T2_slope_estimator.py"]
