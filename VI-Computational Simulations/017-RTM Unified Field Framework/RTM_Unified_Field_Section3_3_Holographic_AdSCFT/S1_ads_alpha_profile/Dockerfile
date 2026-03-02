FROM python:3.10-slim
LABEL description="S1_ads_alpha_profile: RTM Holographic/AdS-CFT"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY S1_ads_alpha_profile.py .
RUN mkdir -p output
CMD ["python", "S1_ads_alpha_profile.py"]
