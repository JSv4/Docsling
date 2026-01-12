# Docling Parser Microservice

This project provides a microservice for parsing PDF documents using the Docling library to extract structured text, annotations, and relationships.

## Features

*   Parses PDF documents (text-based and scanned via OCR).
*   Utilizes Docling and Docling Core for document conversion and analysis.
*   Applies Hierarchical Chunking to group related text items.
*   Generates structured output in OpenContracts format (PAWLS tokens, annotations, relationships).
*   Offers options to force OCR and control relationship grouping.
*   Containerized using Docker with multi-stage builds for development and production.
*   Includes unit and integration tests using `pytest`.

## Technologies Used

*   **Python 3.10+**
*   **Docling & Docling Core:** PDF parsing and structuring engine.
*   **FastAPI:** High-performance web framework.
*   **Pydantic:** Data validation and settings management.
*   **Uvicorn:** ASGI server (for running FastAPI).
*   **Docker & Docker Compose:** Containerization and local orchestration.
*   **Pytest:** Testing framework.
*   **Tesseract & Poppler:** System dependencies for OCR and PDF handling.
*   **Libraries:** `pdfplumber`, `pdf2image`, `pytesseract`, `shapely`, `numpy`, `easyocr`.

## Project Structure

```
.
├── app/                    # Core application logic (parser, models, utils)
│   ├── core/               # Parsing logic, configuration
│   ├── models/             # Pydantic data models (types.py)
│   ├── utils/              # Helper functions (file checks, layout)
│   └── main.py             # FastAPI application entrypoint
├── docling_models/         # Local directory for models (ignored by git, populated by script for local dev)
├── scripts/                # Helper scripts
│   └── download_models.py  # Script to download Docling & EasyOCR models
├── tests/                  # Pytest tests
│   ├── fixtures/           # Test data files (PDF, JSON, TXT)
│   ├── __init__.py
│   ├── conftest.py         # Pytest configuration/fixtures
│   └── test_main.py        # Tests for the FastAPI app
├── .env.example            # Example environment variables file
├── .gitignore
├── .dockerignore
├── Dockerfile              # Multi-stage Dockerfile for building images
├── docker-compose.yml      # Docker Compose file for local development/testing
├── README.md               # This file
├── requirements.txt        # Production Python package dependencies
└── requirements-dev.txt    # Development/testing dependencies
```

## Setup and Installation

### Prerequisites

*   **Python 3.10+** and `pip`.
*   **Docker** and **Docker Compose** (for containerized development/deployment).
*   **Git**.

### Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (for local development/testing without Docker):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    *   For **production runtime** only (usually handled inside Docker build):
      ```bash
      pip install -r requirements.txt
      ```
    *   For **local development or running tests locally**:
      ```bash
      pip install -r requirements-dev.txt
      ```

4.  **Download Models (for Local Development without Docker):**
    *   If you plan to run the FastAPI service locally using `uvicorn` *without* Docker, you need to download the models manually first. Run the download script:
      ```bash
      python scripts/download_models.py --path ./docling_models
      ```
    *   This step is **not** required if you only intend to run the service using Docker/Docker Compose, as the models will be downloaded during the image build process.

5.  **Configure Environment (Optional):**
    *   Copy `.env.example` to `.env`.
    *   Modify `.env` if you need to change the `DOCLING_MODELS_PATH` for local non-Docker development (the default `./docling_models` should work if you ran the download script as above). The path inside Docker containers is handled separately by the `Dockerfile`'s `ENV` instruction.

## Running the Service

### A) Local Development using Docker Compose (Recommended)

This method uses the multi-stage `Dockerfile` (targeting the `builder` stage) and `docker-compose.yml` to create a development environment with hot-reloading. It automatically installs all dependencies (including dev tools) and downloads the models during the image build.

```bash
# Ensure Docker and Docker Compose are running
# From the project root directory:
docker-compose up --build
```
*   This command builds the image (if not already built or if `Dockerfile` changed) targeting the `builder` stage.
*   It starts the service using the command specified in `docker-compose.yml` (`uvicorn` with `--reload`).
*   Your local `./app` directory is mounted into the container, so code changes trigger automatic restarts.
*   Access the service at `http://localhost:8000`.
*   To stop the service, press `Ctrl+C`. To stop and remove the container, run `docker-compose down`.

### B) Running Tests

You can run the `pytest` suite either locally (after installing `requirements-dev.txt`) or within the development container managed by Docker Compose.

```bash
# Option 1: Run tests inside the Docker container (Recommended for consistency)
# Ensure the development container image is built (docker-compose build or docker-compose up --build)
docker-compose run --rm web pytest tests/

# Option 2: Run tests locally
# Ensure you have activated your virtual environment and installed dev dependencies
# source .venv/bin/activate
# pip install -r requirements-dev.txt
pytest -v tests/
```

### C) Building and Running for Production

For production deployment, you build the lean `production` stage of the `Dockerfile` and run it directly using `docker`.

1.  **Build the Production Image:**
    ```bash
    # From the project root directory:
    docker build --target production -t docling-parser-prod:latest .
    ```
    *   `--target production`: Specifies that only the final `production` stage should be built.
    *   `-t docling-parser-prod:latest`: Tags the resulting image for easy reference.

2.  **Run the Production Container:**
    ```bash
    docker run -d -p 8000:8000 --name docling-parser-prod-instance docling-parser-prod:latest
    ```
    *   `-d`: Runs the container in detached mode (in the background).
    *   `-p 8000:8000`: Maps port 8000 on the host to port 8000 in the container.
    *   `--name docling-parser-prod-instance`: Assigns a name to the running container.
    *   `docling-parser-prod:latest`: Specifies the image to run.
    *   You might need to add `--env-file .env.prod` or `-e VAR=value` flags to pass production-specific environment variables.

## Usage

### FastAPI Endpoint (`/parse/`)

*   **Method:** `POST`
*   **URL:** `http://localhost:8000/parse/` (when running locally via Docker Compose or Uvicorn)
*   **Request Body (JSON):**
    ```json
    {
      "filename": "your_document.pdf",
      "pdf_base64": "JVBERi0xLjcKJeLjz9MKMiAwIG...", // Base64 encoded string of the PDF content
      "roll_up_groups": true, // Optional, defaults to false
      "force_ocr": false,     // Optional, defaults to false
      "llm_enhanced_hierarchy": false // Optional, defaults to false
    }
    ```
*   **Success Response (200 OK):** JSON object representing the `OpenContractDocExport` structure.
    ```json
    {
      "title": "...",
      "content": "...",
      // ... other fields ...
    }
    ```
*   **Error Response (400 Bad Request, 422 Unprocessable Entity, 500 Internal Server Error):** Standard FastAPI error JSON.

## Dockerfile Structure (Multi-Stage)

The `Dockerfile` uses a multi-stage build approach to create optimized images for different environments:

1.  **`base` Stage:** Installs Python, common OS packages (like `tesseract`, `poppler`), sets up the working directory, and defines base environment variables.
2.  **`builder` Stage:** Builds upon `base`. Installs *all* Python dependencies from `requirements-dev.txt`, copies the model download script (`scripts/download_models.py`), runs the script to download models, and copies the application (`app/`) and test (`tests/`) code. This stage is used for local development (via `docker-compose.yml`) and running tests in a container.
3.  **`production` Stage:** Builds upon `base`. Installs *only* production Python dependencies from `requirements.txt`. Copies the application code and the downloaded models directly from the `builder` stage. This results in a smaller, more secure image suitable for deployment, excluding development tools and test code.

## Configuration

*   **Docling Models Path:** The path to the Docling models is determined by the `DOCLING_MODELS_PATH` environment variable.
    *   It defaults to `./docling_models` relative to the project root if not set (relevant for local non-Docker execution).
    *   You can set this variable directly or define it in a `.env` file in the project root.
    *   In the `Dockerfile`, this path is set to `/app/docling_models` using an `ENV` instruction, which is used by the download script and the application inside the container.

### Handling Docling Models

The Docling and EasyOCR models required by this service are large and are **not** committed to the Git repository. Instead, they are downloaded using helper functions from the `docling-core` and `easyocr` libraries during the Docker image build process. This is managed by the `scripts/download_models.py` script.

1.  **Download Script:** The `scripts/download_models.py` script uses `docling.pipeline.standard_pdf_pipeline.StandardPdfPipeline.download_models_hf` and `easyocr.Reader` to fetch the necessary models. Ensure the `required_languages` list within the script matches your needs for EasyOCR (default is `['en']`).
2.  **Build Process:** When `docker build` or `docker-compose build` is run, the build process will:
    *   Execute the `scripts/download_models.py` script within the build environment.
    *   The script downloads and saves the models directly into the `/app/docling_models` directory (or the path specified by the `DOCLING_MODELS_PATH` environment variable set within the `Dockerfile`) inside the container image.
3.  **Local Development (Non-Docker):** For local development *without* Docker (e.g., running `uvicorn app.main:app` directly), you must manually download the models first by running the script: `python scripts/download_models.py --path ./docling_models`. The `.gitignore` file prevents this local `docling_models/` directory from being committed.
4.  **Updating Models:** The `docling-core` and `easyocr` libraries manage the model sources. To get updated models, simply rebuild the Docker image (`docker-compose build` or `docker build --target production ...`). This re-runs the download script. If you suspect caching issues or want to ensure the latest versions are fetched, you can add the `--force` flag when the script is called within the `Dockerfile`.
