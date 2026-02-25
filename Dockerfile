FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy both models, server, frontend, and static assets
COPY guard2live_v2_9class.tflite .
COPY guard2live_v3_systemic.tflite .
COPY server.py .
COPY index.html .
COPY static/ static/

# Model paths
ENV V2_MODEL_PATH=guard2live_v2_9class.tflite
ENV V3_MODEL_PATH=guard2live_v3_systemic.tflite
ENV PORT=8000

EXPOSE ${PORT}

CMD ["python", "server.py"]
