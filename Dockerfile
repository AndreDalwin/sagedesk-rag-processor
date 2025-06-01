FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for PyMuPDF
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# The container will start either the web server or the worker
# depending on the CMD argument passed
CMD ["sh", "-c", "if [ \"$PROCESS_TYPE\" = \"worker\" ]; then celery -A celery_app.celery_app worker -l info --concurrency=2; else gunicorn -k uvicorn.workers.UvicornWorker main:app --preload -b 0.0.0.0:$PORT; fi"]
