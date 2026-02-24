FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model, server, and frontend
COPY guard2live_v2_9class.tflite .
COPY server.py .
COPY index.html .

# Render sets PORT automatically; default 8000 for local dev
ENV MODEL_PATH=guard2live_v2_9class.tflite
ENV PORT=8000

EXPOSE ${PORT}

CMD ["python", "server.py"]
