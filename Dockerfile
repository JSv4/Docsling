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
# Sets up Python, common OS dependencies, and working directory.
FROM python:3.10-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
# Path for models inside the container (used by download script and app)
ENV DOCLING_MODELS_PATH=/app/docling_models

# Install common system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    # Add any other common system dependencies here
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# --- Stage 2: Builder ---
# Installs all dependencies (dev + prod), downloads models, copies code.
# Used for development and running tests in container.
FROM base AS builder

# Install build-time OS dependencies (if any are needed only for build/download)
# RUN apt-get update && apt-get install -y --no-install-recommends wget tar && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# Install all Python dependencies (including dev)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-dev.txt

# Copy the model download script
COPY ./scripts/download_models.py /app/scripts/download_models.py

# Download models
# This layer is cached if the script doesn't change. Add --force if needed.
RUN python /app/scripts/download_models.py --path "${DOCLING_MODELS_PATH}"

# Copy application and test code
# Copy app code after installing dependencies and downloading models to leverage cache
COPY ./app /app/app
COPY ./tests /app/tests

# --- Stage 3: Production ---
# Creates the final lean production image.
FROM base AS production

# Copy only production requirements file
COPY requirements.txt ./

# Install only production Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy downloaded models from the builder stage
COPY --from=builder ${DOCLING_MODELS_PATH} ${DOCLING_MODELS_PATH}

# Copy application code from the builder stage
COPY --from=builder /app/app /app/app

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the production application
# Consider using gunicorn for more robust process management in production
# CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 