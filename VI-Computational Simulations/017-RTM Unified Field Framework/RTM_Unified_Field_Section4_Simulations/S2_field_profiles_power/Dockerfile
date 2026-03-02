FROM python:3.10-slim
LABEL description="S2_field_profiles_power: RTM Unified Field Framework Section 4"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S2_field_profiles_power.py .
RUN mkdir -p output
CMD ["python", "S2_field_profiles_power.py"]
