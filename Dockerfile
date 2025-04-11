# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Set the path for Docling models inside the container (can be overridden at runtime)
ENV DOCLING_MODELS_PATH=/app/docling_models

# Set locale for Tesseract if needed (example for UTF-8)
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install system dependencies required by:
# - pdf2image (needs poppler-utils for pdftoppm, pdfinfo)
# - pytesseract (needs tesseract-ocr and potentially language packs like tesseract-ocr-eng)
# - Potentially other libraries (e.g., build tools if installing from source)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    # tesseract-ocr-eng # Add language packs if needed
    poppler-utils \
    # Add any other system dependencies here
    # Clean up apt cache to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Set work directory inside the container
WORKDIR /app

# Copy just the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
# Ensure .dockerignore excludes unnecessary files/folders (like .git, venv, etc.)
COPY ./app /app/app

# Copy Docling models into the image.
# WARNING: This can significantly increase image size.
# For production, consider mounting models as a volume instead.
# Ensure the 'docling_models' directory exists in your build context (where you run 'docker build')
COPY ./docling_models ${DOCLING_MODELS_PATH}
# Optional: Verify models copied correctly (adjust path/check as needed)
# RUN ls -l ${DOCLING_MODELS_PATH}

# Expose the port the app runs on (default for Uvicorn is 8000)
EXPOSE 8000

# Define the command to run the application using Uvicorn
# Use --host 0.0.0.0 to make it accessible from outside the container
# Use --workers for production deployment (e.g., based on CPU cores)
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
# For development/simpler setup:
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 