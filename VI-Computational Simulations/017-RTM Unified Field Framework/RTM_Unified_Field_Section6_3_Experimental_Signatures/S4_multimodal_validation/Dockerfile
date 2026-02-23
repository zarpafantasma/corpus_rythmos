FROM python:3.10-slim
LABEL description="S4_multimodal_validation: RTM Experimental Signatures"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S4_multimodal_validation.py .
RUN mkdir -p output
CMD ["python", "S4_multimodal_validation.py"]
