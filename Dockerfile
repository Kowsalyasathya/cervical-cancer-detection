FROM python:3.9-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Copy project files
COPY . /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# IMPORTANT: Install correct versions
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    tensorflow-cpu==2.10.0 \
    pillow \
    flask \
    gunicorn \
    reportlab

# Expose Render port
EXPOSE 8080

# Start Flask via Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
