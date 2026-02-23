FROM python:3.10-slim
LABEL description="S5_vascular_tree: RTM Unified Field Framework Section 4"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S5_vascular_tree.py .
RUN mkdir -p output
CMD ["python", "S5_vascular_tree.py"]
