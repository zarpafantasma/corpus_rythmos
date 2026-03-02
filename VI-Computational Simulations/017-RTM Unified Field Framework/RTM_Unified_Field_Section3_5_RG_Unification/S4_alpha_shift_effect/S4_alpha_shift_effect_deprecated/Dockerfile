FROM python:3.10-slim
LABEL description="S4_alpha_shift_effect: RTM Unified Field Framework RG Unification"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S4_alpha_shift_effect.py .
RUN mkdir -p output
CMD ["python", "S4_alpha_shift_effect.py"]
