FROM python:3.9-slim

WORKDIR /app

# Install required system packages for TensorFlow
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8080

# Run Gunicorn server
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
