FROM python:3.10-slim
LABEL description="T3_gr_rtm_decomposition: RTM Black Holes Framework"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY T3_gr_rtm_decomposition.py .
RUN mkdir -p output
CMD ["python", "T3_gr_rtm_decomposition.py"]
