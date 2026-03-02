FROM python:3.10-slim
LABEL description="S4_sierpinski_fractal: RTM Unified Field Framework Section 4"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S4_sierpinski_fractal.py .
RUN mkdir -p output
CMD ["python", "S4_sierpinski_fractal.py"]
