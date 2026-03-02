FROM python:3.10-slim
LABEL description="T1_alpha_profile: RTM Black Holes Framework"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY T1_alpha_profile.py .
RUN mkdir -p output
CMD ["python", "T1_alpha_profile.py"]
