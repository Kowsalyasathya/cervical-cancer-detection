FROM python:3.9-slim

WORKDIR /app

# Install system packages for TensorFlow
RUN apt-get update && apt-get install -y \
    libhdf5-serial-dev \
    libatlas-base-dev \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the app
COPY . .

# Ensure models are copied
COPY ./models /app/models

# Expose the web port
EXPOSE 8080

# Start Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
