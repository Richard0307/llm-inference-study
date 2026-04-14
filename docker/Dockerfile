# Base image: official PyTorch with CUDA 12.4 support
# This image already has python + pytorch + cuda runtime installed.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set working directory inside the container
WORKDIR /app

# Copy the Python script into the container
COPY W3_hellodocker.py /app/

# Default command when the container starts
CMD ["python", "W3_hellodocker.py"]
