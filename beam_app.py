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
MEMORY_SIZE = "16Gi"
GPU = "T4" # Add if using GPU-accelerated models, otherwise omit
# PYTHON_VERSION = "python3.10" # Python version now defined by Dockerfile's base image

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Define Model Path ---
# Define the target path inside the container (matches ENV var in Dockerfile)
MODELS_TARGET_PATH = "/app/docling_models" # Ensure this path is accessible/writable

# --- Define Beam Image using Dockerfile ---
image = (
    beam.Image()
    .from_dockerfile(
        path="./Dockerfile", # Path relative to the context root
        context_dir="."
    )
    .add_python_packages(packages="requirements.txt")
)

# --- Loader Function ---
def load_parser_components() -> Optional[Any]:
    """
    Loads and initializes the Docling DocumentConverter once when the container starts.
    """
    logger.info("Executing on_start loader: Initializing DocumentConverter...")
    try:
        # Import necessary components here, within the loader's execution context
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions

        # Ensure the model path exists (should have been created by download script)
        if not os.path.isdir(MODELS_TARGET_PATH):
             logger.error(f"Model directory not found at expected path: {MODELS_TARGET_PATH}")
             # Raise an error to potentially prevent the endpoint from starting incorrectly
             raise FileNotFoundError(f"Model directory not found: {MODELS_TARGET_PATH}")

        # Configure OCR options
        ocr_options = EasyOcrOptions(
            model_storage_directory=MODELS_TARGET_PATH,
            lang=['en'] # Ensure this matches languages downloaded
        )

        # Configure pipeline options
        pipeline_options = PdfPipelineOptions(
            artifacts_path=MODELS_TARGET_PATH,
            do_ocr=True, # Default OCR behavior, can be overridden per-request if needed
            do_table_structure=True,
            generate_page_images=True, # Needed for OCR
            ocr_options=ocr_options,
        )

        # Initialize the converter
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        logger.info("DocumentConverter initialized successfully.")
        # Return the initialized converter (or other necessary components)
        return doc_converter
    except Exception as e:
        logger.exception("Failed to initialize DocumentConverter during on_start.")
        # Get detailed traceback information
        import traceback
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        detailed_traceback = traceback.format_exception(exc_type, exc_value, exc_traceback)
        logger.error(f"Detailed traceback:\n{''.join(detailed_traceback)}")
        
        # Log additional context
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Models path: {MODELS_TARGET_PATH}")
        logger.error(f"Directory exists: {os.path.exists(MODELS_TARGET_PATH)}")
        logger.error(f"Directory is directory: {os.path.isdir(MODELS_TARGET_PATH)}")
        logger.error(f"Directory permissions: {oct(os.stat(MODELS_TARGET_PATH).st_mode)[-3:]}")
        
        # Return None to allow the endpoint to start, but log extensively
        return None

# --- Beam Endpoint Definition ---
@beam.endpoint(
    name=APP_NAME,
    cpu=CPU_COUNT,
    memory=MEMORY_SIZE,
    gpu=GPU, # Uncomment if GPU is needed
    image=image, # Use the image defined above
    keep_warm_seconds=60, # Keep warm for 60 seconds
    on_start=load_parser_components, # Add the loader function here
    timeout=600,
    # volumes=[...] # Keep Volume option commented out unless needed
)
def parse_pdf_beam(context, **inputs: Dict[str, Any]) -> Dict[str, Any]: # Add context
    """
    Beam endpoint to parse a PDF document using Docling.
    Uses a pre-loaded DocumentConverter from the on_start loader.

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
    # --- Retrieve pre-loaded components ---
    # The value returned by load_parser_components is in context.on_start_value
    doc_converter = context.on_start_value

    if doc_converter is None:
         logger.error("DocumentConverter not available from on_start loader. Cannot process request.")
         # Return a 503 Service Unavailable or similar error
         # Note: Beam doesn't directly support setting HTTP status codes in the return dict easily.
         # Returning an error message is standard.
         return {"error": "Parser service initialization failed. Please try again later."}

    # --- Import necessary functions inside the endpoint ---
    try:
        # We only need the internal processing function and the output model now
        from app.core.parser import process_document_with_converter # MODIFIED: Import the new function
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

    # --- Execute Parsing Logic ---
    # Note: We now call the internal function directly with the pre-loaded converter
    try:
        logger.info(f"Processing document {filename} using pre-loaded converter.")
        result_model: Optional[OpenContractDocExport] = process_document_with_converter( # MODIFIED: Call the new function
            doc_converter=doc_converter, # Pass the pre-loaded converter
            pdf_bytes=pdf_bytes,
            pdf_filename=filename,
            force_ocr=force_ocr, # Pass options
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

    except Exception as e:
        # Catch errors during the _internal_process_document call
        logger.exception(f"Unexpected error during PDF processing for {filename}: {e}")
        return {"error": f"An unexpected internal server error occurred during processing: {type(e).__name__}"} 