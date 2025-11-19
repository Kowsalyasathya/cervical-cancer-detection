# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system deps for Pillow and reportlab if needed
RUN apt-get update && apt-get install -y build-essential libjpeg-dev zlib1g-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT 8080
EXPOSE 8080

# Use gunicorn for production
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
