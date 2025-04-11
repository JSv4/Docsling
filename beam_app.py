import beam
import base64
import io
import os
import logging
from typing import Optional, Dict, Any

# --- Configuration ---
# Adjust resource requirements as needed. PDF processing can be intensive.
APP_NAME = "docling-parser"
CPU_COUNT = 4
MEMORY_SIZE = "8Gi"
# GPU = "T4" # Add if using GPU-accelerated models, otherwise omit
PYTHON_VERSION = "python3.10" # Match your project's Python version

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Define Beam Image ---
# Replicate dependencies from requirements.txt and Dockerfile system installs
image = beam.Image(
    python_version=PYTHON_VERSION,
    python_packages=[
        "fastapi", # Keep if models depend on it, otherwise optional
        "uvicorn[standard]", # Optional for beam app
        "pydantic>=2.0.0,<3.0.0", # Ensure version compatibility
        "python-multipart", # Likely not needed for base64 input
        # Docling dependencies
        "docling",
        "docling-core",
        # Other parser dependencies
        "numpy",
        "pdf2image",
        "python-pytesseract",
        "pdfplumber",
        "shapely",
        # Add any other specific versions if needed
    ],
    apt_packages=[
        "tesseract-ocr", # For pytesseract
        # "tesseract-ocr-eng", # Add specific language packs if needed
        "poppler-utils", # For pdf2image
    ],
    commands=[
        "pip install --upgrade pip",
        # Add any other setup commands if required
    ],
)

# --- Beam Endpoint Definition ---
@beam.endpoint(
    name=APP_NAME,
    cpu=CPU_COUNT,
    memory=MEMORY_SIZE,
    # gpu=GPU, # Uncomment if GPU is needed
    image=image,
    keep_warm_seconds=60, # Optional: reduce cold starts
    # Volumes can be mounted here if models are stored externally
    # volumes=[beam.Volume(name="docling_models_vol", mount_path="/mnt/models")]
)
def parse_pdf_beam(**inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Beam endpoint to parse a PDF document using Docling.

    Expects input JSON with:
    - pdf_base64 (str): Base64 encoded content of the PDF file.
    - filename (str): The original filename (used for logging/metadata).
    - force_ocr (bool, optional): Force OCR processing. Defaults to False.
    - roll_up_groups (bool, optional): Roll up groups. Defaults to False.
    - llm_enhanced_hierarchy (bool, optional): Use LLM enhancement. Defaults to False.

    Returns:
        A dictionary containing the parsed 'result' (OpenContractDocExport structure)
        or an 'error' message.
    """
    # --- Import necessary functions inside the endpoint ---
    # This ensures imports happen within the Beam execution environment
    # Note: Ensure the 'app' directory is included in the deployment package
    try:
        from app.core.parser import process_document_dynamic_init
        from app.models.types import OpenContractDocExport
    except ImportError as e:
         logger.exception("Failed to import application modules within Beam endpoint.")
         return {"error": f"Internal server error: Failed to import modules - {e}"}

    # --- Input Validation ---
    pdf_base64 = inputs.get("pdf_base64")
    filename = inputs.get("filename")

    if not pdf_base64 or not isinstance(pdf_base64, str):
        return {"error": "Missing or invalid 'pdf_base64' input (must be a base64 string)."}
    if not filename or not isinstance(filename, str):
        return {"error": "Missing or invalid 'filename' input (must be a string)."}

    # --- Decode PDF ---
    try:
        pdf_bytes = base64.b64decode(pdf_base64)
        if not pdf_bytes:
            return {"error": "Decoded PDF content is empty."}
        logger.info(f"Received request for file: {filename}, {len(pdf_bytes)} bytes decoded.")
    except Exception as e:
        logger.error(f"Failed to decode base64 PDF content for {filename}: {e}")
        return {"error": f"Invalid base64 encoding for PDF: {e}"}

    # --- Get Options ---
    force_ocr = inputs.get("force_ocr", False)
    roll_up_groups = inputs.get("roll_up_groups", False)
    llm_enhanced_hierarchy = inputs.get("llm_enhanced_hierarchy", False)
    logger.info(f"Processing options: force_ocr={force_ocr}, roll_up_groups={roll_up_groups}, llm_enhanced_hierarchy={llm_enhanced_hierarchy}")

    # --- Define Model Path within Beam Container ---
    # Beam copies the deployment directory to /app inside the container.
    # If 'docling_models' is in the root of your deployment dir, this path should work.
    models_path_in_container = "/app/docling_models"
    logger.info(f"Expecting Docling models at: {models_path_in_container}")

    # --- Execute Parsing Logic ---
    try:
        result_model: Optional[OpenContractDocExport] = process_document_dynamic_init(
            pdf_bytes=pdf_bytes,
            pdf_filename=filename,
            models_path=models_path_in_container, # Pass the path here
            force_ocr=force_ocr,
            roll_up_groups=roll_up_groups,
            llm_enhanced_hierarchy=llm_enhanced_hierarchy,
        )

        # --- Format Output ---
        if result_model:
            logger.info(f"Successfully processed {filename}. Returning result.")
            # Use .model_dump() for Pydantic v2, .dict() for v1
            # mode='json' ensures JSON-serializable types (like enums converted to strings)
            return {"result": result_model.model_dump(mode='json')}
        else:
            logger.error(f"Processing function returned None for {filename}.")
            # Provide a more specific error if possible based on logs from process_document
            return {"error": "Failed to process document. Check service logs for details."}

    except FileNotFoundError as fnf_error:
         # Catch model path errors specifically if they bubble up
         logger.exception(f"Model path error during processing for {filename}: {fnf_error}")
         return {"error": f"Internal server error: Could not find models at expected path - {fnf_error}"}
    except Exception as e:
        logger.exception(f"Unexpected error during PDF processing for {filename}: {e}")
        # Avoid leaking detailed stack traces in the response unless desired
        return {"error": f"An unexpected internal server error occurred: {type(e).__name__}"} 