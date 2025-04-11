import io
import logging
import pdfplumber # Ensure this is in requirements.txt

logger = logging.getLogger(__name__)

def check_if_pdf_needs_ocr(pdf_stream: io.BytesIO) -> bool:
    """
    Checks if a PDF likely requires OCR by attempting to extract text using pdfplumber.

    It checks the first few pages for any significant text content. If no text is
    found, it suggests that OCR is needed.

    Args:
        pdf_stream: A file-like object (BytesIO) containing the PDF data.
                    The stream position will be reset to 0 after checking.

    Returns:
        True if OCR is likely needed (no text found or error during check), False otherwise.
    """
    initial_position = pdf_stream.tell()
    try:
        with pdfplumber.open(pdf_stream) as pdf:
            if not pdf.pages:
                logger.warning("PDF has no pages. Assuming OCR might be needed or file is invalid.")
                return True # Treat as needing OCR or potentially invalid

            # Check the first few pages for text content
            pages_to_check = min(len(pdf.pages), 3)
            has_text = False
            for i in range(pages_to_check):
                try:
                    page_text = pdf.pages[i].extract_text(x_tolerance=2, y_tolerance=2) # Adjust tolerances if needed
                    # Check for meaningful text content, not just whitespace or noise
                    if page_text and len(page_text.strip()) > 10: # Heuristic: check for more than 10 non-whitespace chars
                        has_text = True
                        logger.info(f"Text found on page {i+1} using pdfplumber.")
                        break
                except Exception as page_error:
                    logger.warning(f"Could not extract text from page {i+1} with pdfplumber: {page_error}")
                    continue # Try next page

            if not has_text:
                logger.info("No significant text found in the first %d pages using pdfplumber. OCR is likely needed.", pages_to_check)
                return True
            else:
                logger.info("Text found using pdfplumber. OCR might not be needed.")
                return False
    except Exception as e:
        # Includes pdfplumber.PDFSyntaxError etc.
        logger.error(f"Error checking PDF with pdfplumber: {e}. Assuming OCR is needed.", exc_info=True)
        return True # Assume OCR needed if pdfplumber fails to open/read
    finally:
        # Reset stream position for subsequent use
        pdf_stream.seek(initial_position) 