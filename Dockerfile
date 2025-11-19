# Use Ubuntu base
FROM ubuntu:22.04

# Avoid timezone prompt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python & needed system dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev && \
    apt-get clean

# Set python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Set work directory
WORKDIR /app

# Copy project files
COPY . /app

# Install required Python packages
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir flask gunicorn pillow numpy tensorflow-cpu==2.10.0 reportlab

# Expose Render's default port
EXPOSE 8080

# Run the app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
