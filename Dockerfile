# Use Python runtime
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for Pillow + TensorFlow
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Render port
EXPOSE 8080

# Start Flask using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
