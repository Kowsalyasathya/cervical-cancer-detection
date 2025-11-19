# Use Python 3.9
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Copy models folder (important!)
COPY ./models /app/models

# Expose port
EXPOSE 8080

# Start app
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
