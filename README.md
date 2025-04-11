# Docling Parser Microservice

This project provides a microservice for parsing PDF documents using the Docling library to extract structured text, annotations, and relationships. It offers two deployment options: a persistent FastAPI web service and a serverless Beam endpoint.

## Features

*   Parses PDF documents (text-based and scanned via OCR).
*   Utilizes Docling and Docling Core for document conversion and analysis.
*   Applies Hierarchical Chunking to group related text items.
*   Generates structured output in OpenContracts format (PAWLS tokens, annotations, relationships).
*   Offers options to force OCR and control relationship grouping.
*   Includes both a FastAPI web server implementation and a Beam serverless endpoint.
*   Containerized using Docker.
*   Includes unit and integration tests using `pytest`.

## Technologies Used

*   **Python 3.10+**
*   **Docling & Docling Core:** PDF parsing and structuring engine.
*   **FastAPI:** High-performance web framework (for the web service option).
*   **Beam:** Serverless platform (for the serverless endpoint option).
*   **Pydantic:** Data validation and settings management.
*   **Uvicorn:** ASGI server (for running FastAPI).
*   **Docker & Docker Compose:** Containerization and local orchestration.
*   **Pytest:** Testing framework.
*   **Tesseract & Poppler:** System dependencies for OCR and PDF handling.
*   **Libraries:** `pdfplumber`, `pdf2image`, `pytesseract`, `shapely`, `numpy`.

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
│   ├── conftest.py         # Pytest configuration/fixtures (if any)
│   ├── test_main.py        # Tests for the FastAPI app (app/main.py)
│   └── test_beam_app.py    # Tests for the Beam endpoint function (beam_app.py)
├── .env.example            # Example environment variables file
├── .gitignore
├── .dockerignore
├── beam_app.py             # Beam endpoint definition script
├── Dockerfile              # Dockerfile for building the service image
├── docker-compose.yml      # Docker Compose file for local development/testing
├── README.md               # This file
└── requirements.txt        # Python package dependencies
```

## Setup and Installation

### Prerequisites

*   **Python 3.10+** and `pip`.
*   **Docker** and **Docker Compose** (for containerized deployment/testing).
*   **Tesseract OCR Engine:** Install system-wide. (e.g., `sudo apt-get install tesseract-ocr tesseract-ocr-eng` on Debian/Ubuntu, `brew install tesseract` on macOS).
*   **Poppler Utilities:** Install system-wide (provides `pdftoppm`, `pdfinfo`). (e.g., `sudo apt-get install poppler-utils` on Debian/Ubuntu, `brew install poppler` on macOS).
*   **Beam CLI** (Optional, only if deploying/testing the Beam endpoint): `pip install beam-cli` and configure with `beam configure`.

### Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Models (for Local Development without Docker):**
    *   If you plan to run the FastAPI service locally using `uvicorn` *without* Docker, you need to download the models manually first. Run the download script:
      ```bash
      python scripts/download_models.py --path ./docling_models
      ```
    *   This step is **not** required if you only intend to run the service using Docker/Docker Compose or deploy via Beam, as the models will be downloaded during the image build process.

5.  **Configure Environment (Optional):**
    *   Copy `.env.example` to `.env`.
    *   Modify `.env` if you need to change the `DOCLING_MODELS_PATH` for local non-Docker development (the default `./docling_models` should work if you ran the download script as above). The path inside Docker/Beam containers is handled separately.

## Running the Service

You can run the parser either as a persistent FastAPI service or deploy it as a serverless Beam endpoint.

### Option 1: Running the FastAPI Service

#### A) Locally with Uvicorn (for development)

Make sure you have installed prerequisites and dependencies locally.

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`, with interactive documentation at `http://localhost:8000/docs`.

#### B) Using Docker Compose (Recommended for local testing/dev)

This method uses the `Dockerfile` and `docker-compose.yml` to build the image and run the container, including system dependencies. **The image build process automatically runs the `scripts/download_models.py` script to download the required models into the image.**

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`. To stop the service, press `Ctrl+C` and then run `docker-compose down`.

### Option 2: Deploying the Beam Endpoint

This option deploys the parsing logic as a serverless function on the Beam platform.

1.  **Prerequisites:** Ensure Beam CLI is installed and configured.
2.  **Deploy:** Navigate to the project root directory (containing `beam_app.py`, `app/`, `scripts/`) and run:
    ```bash
    beam deploy beam_app.py
    ```
    Beam will package the code (including the `scripts/` directory), build the image (which involves running the `scripts/download_models.py` script inside the build environment), and deploy the endpoint.

## Usage

### FastAPI Endpoint (`/parse/`)

*   **Method:** `POST`
*   **URL:** `http://<host>:<port>/parse/` (e.g., `http://localhost:8000/parse/`)
*   **Request Type:** `multipart/form-data`
*   **Form Fields:**
    *   `file`: The PDF file to parse (Required).
    *   `force_ocr` (boolean, Optional): Force OCR even if text is detected. Defaults to `false`.
    *   `roll_up_groups` (boolean, Optional): Roll up items under the same heading into single relationships. Defaults to `false`.
    *   `llm_enhanced_hierarchy` (boolean, Optional): Apply experimental LLM hierarchy enhancement (Placeholder). Defaults to `false`.
*   **Success Response (200 OK):** JSON object matching the `OpenContractDocExport` Pydantic model (see `app/models/types.py`).
*   **Error Responses:** Standard FastAPI HTTP errors (e.g., 415 for unsupported media type, 422 for validation errors, 500 for internal server errors, 503 if service is unavailable).

**Example using `curl`:**

```bash
curl -X POST "http://localhost:8000/parse/" \
  -F "file=@/path/to/your/document.pdf" \
  -F "roll_up_groups=true" \
  -F "force_ocr=false"
```

### Beam Endpoint (`docling-parser`)

*   **Invocation:** Use Beam SDK, `curl`, or other HTTP clients to send a POST request to the deployed endpoint URL provided by Beam.
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
*   **Success Response:** JSON object containing a `result` key with the `OpenContractDocExport` structure.
    ```json
    {
      "result": {
        "title": "...",
        "content": "...",
        // ... other fields ...
      }
    }
    ```
*   **Error Response:** JSON object containing an `error` key with a description of the issue.
    ```json
    {
      "error": "Missing or invalid 'pdf_base64' input (must be a base64 string)."
    }
    ```

## Running Tests

The project uses `pytest` for testing both the FastAPI application and the Beam endpoint function logic.

1.  Ensure you have installed development dependencies (including `pytest` and `httpx`).
2.  Make sure the test fixture PDF (`tests/fixtures/...Development_Agreement_ZrZJLLv.pdf`) exists.
3.  Run tests from the project root directory:

    ```bash
    pytest -v tests/
    ```
    *   `tests/test_main.py` contains tests for the FastAPI endpoints using `TestClient`.
    *   `tests/test_beam_app.py` contains tests for the `parse_pdf_beam` function by calling it directly and simulating inputs.

## Configuration

*   **Docling Models Path:** The path to the Docling models is determined by the `DOCLING_MODELS_PATH` environment variable.
    *   It defaults to `./docling_models` relative to the project root if not set.
    *   You can set this variable directly or define it in a `.env` file in the project root.
    *   In the `Dockerfile` and `beam_app.py`, this path is typically set/expected to be `/app/docling_models` inside the container/runtime environment.

### Handling Docling Models

The Docling and EasyOCR models required by this service are large and are **not** committed to the Git repository. Instead, they are downloaded using helper functions from the `docling-core` and `easyocr` libraries during the Docker image build process (for both FastAPI and Beam deployments). This is managed by the `scripts/download_models.py` script.

1.  **Download Script:** The `scripts/download_models.py` script uses `docling.pipeline.standard_pdf_pipeline.StandardPdfPipeline.download_models_hf` and `easyocr.Reader` to fetch the necessary models. Ensure the `required_languages` list within the script matches your needs for EasyOCR (default is `['en']`).
2.  **Build Process:** When `docker build`, `docker-compose build`, or `beam deploy` is run, the build process will:
    *   Execute the `scripts/download_models.py` script within the build environment.
    *   The script downloads and saves the models directly into the `/app/docling_models` directory (or the path specified by the `DOCLING_MODELS_PATH` environment variable set within the `Dockerfile`) inside the container image.
3.  **Local Development (Non-Docker):** For local development *without* Docker (e.g., running `uvicorn app.main:app` directly), you must manually download the models first by running the script: `python scripts/download_models.py --path ./docling_models`. The `.gitignore` file prevents this local `docling_models/` directory from being committed.
4.  **Updating Models:** The `docling-core` and `easyocr` libraries manage the model sources. To get updated models, simply rebuild the Docker image (`docker-compose build` or `docker build ...`) or redeploy the Beam app (`beam deploy beam_app.py`). This re-runs the download script. If you suspect caching issues or want to ensure the latest versions are fetched, you can add the `--force` flag when the script is called within the `Dockerfile` or `beam_app.py`'s `commands` list (e.g., `python /app/scripts/download_models.py --path "${DOCLING_MODELS_PATH}" --force`). 