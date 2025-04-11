import logging
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, status
from fastapi.responses import JSONResponse

# Import core processing logic and models
from app.core.parser import process_document, doc_converter # Import doc_converter to check status
from app.models.types import OpenContractDocExport
from app.core.config import settings # Ensure settings are loaded

# Configure logging for FastAPI/Uvicorn
# Uvicorn usually handles logging config, but basicConfig can be a fallback
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Docling Parser Service",
    description="Microservice to parse PDF documents using Docling and extract structured data.",
    version="0.1.0",
)

# --- Dependency for checking service readiness ---
async def check_converter_ready():
    """Dependency that checks if the Docling converter is initialized."""
    if doc_converter is None:
        logger.error("Docling converter dependency check failed: Converter not initialized.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not ready: Docling converter failed to initialize."
        )
    # Could add more checks here if needed (e.g., model files loaded)
    logger.debug("Docling converter dependency check passed.")


# --- API Endpoints ---

@app.post(
    "/parse/",
    response_model=OpenContractDocExport, # Use the Pydantic model for response validation and serialization
    summary="Parse PDF Document",
    description="Upload a PDF file, process it using Docling, and return structured annotations, relationships, and text.",
    tags=["Parsing"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error during processing"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid input (e.g., empty file, bad filename)"},
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"description": "File type not supported (only PDF)"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Service is not ready (e.g., models not loaded)"},
    }
)
async def parse_pdf_endpoint(
    file: UploadFile = File(..., description="The PDF file to parse."),
    force_ocr: bool = Form(False, description="Force OCR processing even if text is detectable by pdfplumber."),
    roll_up_groups: bool = Form(False, description="Roll up items under the same heading into single relationships."),
    llm_enhanced_hierarchy: bool = Form(False, description="Apply experimental LLM-based hierarchy enhancement (Placeholder)."),
    # Check converter readiness before processing the request
    converter_ready: None = Depends(check_converter_ready)
):
    """
    Receives a PDF file and processing options, orchestrates the parsing
    using the core `process_document` function, and returns the structured
    result or raises appropriate HTTP errors.
    """
    logger.info(f"Received request to parse file: {file.filename} (Content-Type: {file.content_type})")
    logger.info(f"Parsing options: force_ocr={force_ocr}, roll_up_groups={roll_up_groups}, llm_enhanced_hierarchy={llm_enhanced_hierarchy}")

    # --- Input Validation ---
    if not file.filename:
         logger.warning("Upload request rejected: Filename not provided.")
         raise HTTPException(
             status_code=status.HTTP_400_BAD_REQUEST,
             detail="Filename must be provided."
         )
    # Allow common PDF MIME types
    allowed_mime_types = ["application/pdf", "application/x-pdf"]
    if file.content_type not in allowed_mime_types:
        logger.warning(f"Upload request rejected: Unsupported file type '{file.content_type}'.")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: '{file.content_type}'. Only PDF ({', '.join(allowed_mime_types)}) is allowed."
        )

    try:
        # Read file content into memory
        pdf_bytes = await file.read()
        if not pdf_bytes:
             logger.warning(f"Upload request rejected: Received empty file for '{file.filename}'.")
             raise HTTPException(
                 status_code=status.HTTP_400_BAD_REQUEST,
                 detail="Received empty file."
             )
        logger.info(f"Read {len(pdf_bytes)} bytes from uploaded file '{file.filename}'.")

        # --- Call Core Processing Logic ---
        result = process_document(
            pdf_bytes=pdf_bytes,
            pdf_filename=file.filename, # Pass filename for logging/metadata
            force_ocr=force_ocr,
            roll_up_groups=roll_up_groups,
            llm_enhanced_hierarchy=llm_enhanced_hierarchy,
        )

        # --- Handle Processing Outcome ---
        if result is None:
            # process_document returns None on critical internal errors
            logger.error(f"Processing failed for file '{file.filename}'. Returning 500.")
            # Avoid leaking detailed internal errors to the client unless necessary
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process the document due to an internal server error."
            )

        # Return the successful result (FastAPI handles Pydantic model serialization to JSON)
        logger.info(f"Successfully processed '{file.filename}'. Returning result.")
        # FastAPI automatically uses the `response_model` for serialization, including aliases.
        return result

    except HTTPException as http_exc:
         # Re-raise exceptions that are already HTTPExceptions
         raise http_exc
    except Exception as e:
        # Catch any other unexpected errors during file reading or processing setup
        logger.error(f"Unexpected error during request handling for '{file.filename}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while handling the request: {str(e)}"
        )
    finally:
        # Ensure the uploaded file resource is closed
        await file.close()
        logger.debug(f"Closed file handle for '{file.filename}'.")


@app.get(
    "/health",
    summary="Health Check",
    description="Checks if the service is running and the core Docling converter is initialized.",
    tags=["Health"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Service is not ready (converter not initialized)"}
    }
)
async def health_check(converter_ready: None = Depends(check_converter_ready)):
    """
    Basic health check endpoint.
    Relies on the `check_converter_ready` dependency to ensure the service
    is in a usable state.
    """
    return {"status": "ok", "message": "Docling Parser Service is running and converter is ready."}

# --- Optional: Add startup/shutdown events if needed ---
# @app.on_event("startup")
# async def startup_event():
#     logger.info("Docling Parser Service starting up...")
#     # Perform any initialization here that isn't handled by global scope
#
# @app.on_event("shutdown")
# async def shutdown_event():
#     logger.info("Docling Parser Service shutting down...")
#     # Perform any cleanup here 