FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy both models, server, and frontend
COPY guard2live_v2_9class.tflite .
COPY guard2live_v3_systemic.tflite .
COPY server.py .
COPY index.html .

# Model paths
ENV V2_MODEL_PATH=guard2live_v2_9class.tflite
ENV V3_MODEL_PATH=guard2live_v3_systemic.tflite
ENV PORT=8000

EXPOSE ${PORT}

CMD ["python", "server.py"]
