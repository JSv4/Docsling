import logging
import base64
import json
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.parser import process_document_dynamic_init # Import the dynamic init function
from app.core.config import settings # Import settings to get models path
from app.models.types import OpenContractDocExport

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

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "message": "Docling Parser Service is running and converter is ready."}

@app.post("/parse/", response_model=OpenContractDocExport)
async def parse_pdf(request: Request):
    """
    Parse a PDF document and return structured data.
    
    Accepts a base64-encoded PDF and returns an OpenContractDocExport object.
    """
    try:
        # Check content type for file uploads
        content_type = request.headers.get("content-type", "")
        
        # Handle form data (file upload)
        if "multipart/form-data" in content_type:
            form = await request.form()
            if "file" not in form:
                raise HTTPException(status_code=422, detail="No file part in the request")
            
            file = form["file"]
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=415, detail="Unsupported Media Type. Only PDF files are accepted.")
            
            # Read file content
            pdf_bytes = await file.read()
            filename = file.filename
            
            # Extract options from form data
            force_ocr = form.get("force_ocr", "false").lower() == "true"
            roll_up_groups = form.get("roll_up_groups", "false").lower() == "true"
            llm_enhanced_hierarchy = form.get("llm_enhanced_hierarchy", "false").lower() == "true"
        # Handle JSON request
        elif "application/json" in content_type:
            try:
                data = await request.json()
                pdf_base64 = data.get("pdf_base64")
                if not pdf_base64:
                    raise HTTPException(status_code=400, detail="Missing pdf_base64 field")
                
                pdf_bytes = base64.b64decode(pdf_base64)
                filename = data.get("filename", "document.pdf")
                
                # Extract options
                force_ocr = data.get("force_ocr", False)
                roll_up_groups = data.get("roll_up_groups", False)
                llm_enhanced_hierarchy = data.get("llm_enhanced_hierarchy", False)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 encoding for PDF: {e}")
        # If neither multipart/form-data nor application/json
        else:
            raise HTTPException(
                status_code=422,
                detail="Unsupported content type. Use multipart/form-data for file upload or application/json for base64 encoded PDF."
            )
            
        logger.info(f"Received parse request for file: {filename}")
        logger.info(f"Passing options to parser - force_ocr: {force_ocr}, roll_up_groups: {roll_up_groups}, llm_enhanced_hierarchy: {llm_enhanced_hierarchy}")
        
        # Call the dynamic initialization processing function
        result_model = process_document_dynamic_init(
            pdf_bytes=pdf_bytes,
            pdf_filename=filename,
            models_path=settings.DOCLING_MODELS_PATH,
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
    except Exception as e:
        # If it's already an HTTPException, re-raise it
        if isinstance(e, HTTPException):
            raise
        # Otherwise, wrap it in a 400 Bad Request
        logger.error(f"Unexpected error processing request: {e}")
        raise HTTPException(status_code=400, detail=str(e)) 