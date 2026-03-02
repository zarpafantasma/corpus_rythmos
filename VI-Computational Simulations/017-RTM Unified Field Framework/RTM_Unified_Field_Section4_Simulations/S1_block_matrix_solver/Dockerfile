FROM python:3.10-slim
LABEL description="S1_block_matrix_solver: RTM Unified Field Framework Section 4"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S1_block_matrix_solver.py .
RUN mkdir -p output
CMD ["python", "S1_block_matrix_solver.py"]
