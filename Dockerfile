# ==============================================================================
# Build and Run Instructions
# ==============================================================================
#
# --- Development ---
# Purpose: Run locally with hot-reloading and all dev dependencies.
# Build: Handled automatically by docker-compose.
# Run:   docker-compose up --build
# Notes: Uses the 'builder' stage implicitly via docker-compose.yml target.
#        Mounts local code for hot-reloading.
#
# --- Testing ---
# Purpose: Run tests within a containerized environment.
# Build: Handled automatically by docker-compose.
# Run:   docker-compose run --rm web pytest tests/
# Notes: Uses the 'builder' stage image built by compose. '--rm' removes container after test run.
#        Alternatively, run tests locally after `pip install -r requirements-dev.txt`.
#
# --- Production ---
# Purpose: Build a lean image with only runtime dependencies for deployment.
# Build:   docker build --target production -t docling-parser-prod:latest .
# Run:     docker run -d -p 8000:8000 --name docling-parser docling-parser-prod:latest
# Notes: Builds *only* the final 'production' stage. Does not include dev tools or hot-reloading.
#
# ==============================================================================

# --- Stage 1: Final Image ---
# Use official PyTorch image with CUDA 12.4 / cuDNN 9
# Use the fully qualified name
FROM docker.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS final

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
# Path for models inside the container
ENV DOCLING_MODELS_PATH=/app/docling_models

# Install essential tools, pip, and required system dependencies
# Ensure pip and venv tools are present for the base image's python3
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    tesseract-ocr \
    poppler-utils \
    build-essential \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Add any other common system dependencies here
    && rm -rf /var/lib/apt/lists/* \
    # Ensure pip is upgraded for the default python3
    && python3 -m pip install --no-cache-dir --upgrade pip

# Set work directory
WORKDIR /app

# Copy requirements file FIRST to leverage Docker layer caching
# Use simple source path, Docker build context root is assumed
COPY requirements.txt ./

# Install Python dependencies GLOBALLY into the main environment
# This avoids conflicts with system packages vs --user packages
RUN echo "---- Contents of requirements.txt seen by Docker build ----" && \
    cat requirements.txt && \
    echo "---- Attempting to install dependencies... ----" && \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    echo "---- Installation complete. Installed packages (pip freeze): ----" && \
    python3 -m pip freeze && \
    echo "---- Checking for broken dependencies (pip check): ----" && \
    python3 -m pip check && \
    echo "---- Dependency check complete. ----"

# Copy the model download script
# Use simple source path
COPY scripts/download_models.py /app/scripts/download_models.py

# Download models (needs to happen after dependencies like easyocr are installed)
RUN python3 /app/scripts/download_models.py --path "${DOCLING_MODELS_PATH}"

# Copy application code (can be done last)
# Use simple source path
COPY app /app/app

# Expose the port the app runs on (for standard FastAPI, not directly used by Beam endpoint)
EXPOSE 8000

# Define a default command (useful for running the container directly, not used by Beam endpoint)
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Beam uses this image environment but runs code from /mnt/code ---
# The key is that the Python environment itself is now consistent. 