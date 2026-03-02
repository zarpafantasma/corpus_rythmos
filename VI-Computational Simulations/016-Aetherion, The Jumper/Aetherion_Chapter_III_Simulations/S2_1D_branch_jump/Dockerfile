FROM python:3.10-slim
LABEL description="S2_1D_branch_jump: Aetherion Chapter III Simulation"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S2_1D_branch_jump.py .
RUN mkdir -p output
CMD ["python", "S2_1D_branch_jump.py"]
