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

# --- Stage 1: Base ---
# Use official PyTorch image with CUDA 12.4 / cuDNN 9
# Use the fully qualified name for compatibility with Beam's build system
FROM docker.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
# Path for models inside the container (used by download script and app)
ENV DOCLING_MODELS_PATH=/app/docling_models
# Add Python's user site-packages bin to PATH
# Assuming the default python3 in the image is 3.11
ENV PATH="/root/.local/bin:${PATH}"
# Explicitly tell Python where to find packages installed with --user
# --- ADJUST PYTHON VERSION IN PATH ---
ENV PYTHONPATH="/root/.local/lib/python3.11/site-packages:${PYTHONPATH:-}"

# Install essential tools, pip, and required system dependencies
# --- REMOVE python3.10 INSTALLATION ---
# The PyTorch image provides Python (likely 3.11). Ensure pip and venv tools are present.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    tesseract-ocr \
    poppler-utils \
    # Add any other common system dependencies here (e.g., git, curl if needed)
    && rm -rf /var/lib/apt/lists/* \
    # Ensure pip is upgraded for the default python3
    && python3 -m pip install --upgrade pip

# Set work directory
WORKDIR /app

# --- Stage 2: Builder ---
# Installs remaining dependencies, downloads models, copies code.
FROM base AS builder

# Copy requirements file
COPY requirements.txt ./

# Install remaining Python dependencies from requirements.txt using the default python3
# Pip should detect the pre-installed torch and skip it if versions match.
RUN python3 -m pip install --no-cache-dir --user -r requirements.txt # Removed --upgrade pip here as it's done in base

# Copy the model download script
COPY ./scripts/download_models.py /app/scripts/download_models.py

# Download models
RUN python3 /app/scripts/download_models.py --path "${DOCLING_MODELS_PATH}"

# Copy application and test code
COPY ./app /app/app
# COPY ./tests /app/tests # Optional: only needed if running tests in builder <-- REMOVE OR COMMENT OUT

# --- Stage 3: Production ---
# Creates the final lean production image.
# Inherits PyTorch, Python 3.11, Tesseract, Poppler from the 'base' stage.
FROM base AS production

# Copy installed Python packages (excluding torch, already in base) from the builder stage
COPY --from=builder /root/.local /root/.local

# Copy downloaded models from the builder stage
COPY --from=builder ${DOCLING_MODELS_PATH} ${DOCLING_MODELS_PATH}

# Copy application code from the builder stage
COPY --from=builder /app/app /app/app

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the production application using the default python3
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 