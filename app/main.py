import logging
import base64
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.parser import process_document_dynamic_init # Import the dynamic init function
from app.core.config import settings # Import settings to get models path
from app.models.types import ParseRequest, OpenContractDocExport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Docling Parser Service",
    description="API for parsing PDF documents using Docling",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Error Handling ---

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )

# --- Routes ---

@app.get("/")
async def root():
    """Root endpoint for health checks."""
    return {"status": "ok", "message": "Docling Parser Service is running"}

@app.post("/parse/", response_model=OpenContractDocExport)
async def parse_pdf(request: ParseRequest):
    """
    Parse a PDF document and return structured data.
    
    Accepts a base64-encoded PDF and returns an OpenContractDocExport object.
    """
    logger.info(f"Received parse request for file: {request.filename}")
    
    # Decode the base64 PDF content
    try:
        pdf_base64 = request.pdf_base64
        pdf_bytes = base64.b64decode(pdf_base64)
        if not pdf_bytes:
            logger.error("Decoded PDF content is empty.")
            raise HTTPException(status_code=400, detail="Decoded PDF content is empty.")
        logger.info(f"Successfully decoded {len(pdf_bytes)} bytes of PDF data.")
    except Exception as e:
        logger.error(f"Failed to decode base64 PDF content: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid base64 encoding for PDF: {e}")
    
    # Extract parameters
    filename = request.filename
    force_ocr = request.force_ocr
    roll_up_groups = request.roll_up_groups
    llm_enhanced_hierarchy = request.llm_enhanced_hierarchy
    
    # Call the dynamic initialization processing function
    result_model = process_document_dynamic_init(
        pdf_bytes=pdf_bytes,
        pdf_filename=filename,
        models_path=settings.DOCLING_MODELS_PATH, # Pass the models path from settings
        force_ocr=force_ocr,
        roll_up_groups=roll_up_groups,
        llm_enhanced_hierarchy=llm_enhanced_hierarchy,
    )
    
    if result_model:
        logger.info(f"Successfully processed {filename}. Returning result.")
        return result_model
    else:
        logger.error(f"Failed to process {filename}. Returning 500 error.")
        raise HTTPException(
            status_code=500,
            detail="Failed to process document. Check server logs for details."
        ) 