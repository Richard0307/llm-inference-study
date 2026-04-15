# Base image: official PyTorch with CUDA 12.4 support
# This image already has python + pytorch + cuda runtime installed.
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
# Copy the Python script into the container
COPY W3_hellodocker.py /app/

# Default command when the container starts
CMD ["python", "W3_hellodocker.py"]
