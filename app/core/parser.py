import io
import logging
import os
import traceback
from io import BytesIO
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, cast

import numpy as np
import pdf2image
import pytesseract
from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import ListItem
from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    SectionHeaderItem,
    TextItem,
    PageItem as DoclingPage, # Explicitly import Docling's Page
)
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from shapely.geometry import box
from shapely.strtree import STRtree

# Import from our microservice structure
from app.models.types import (
    OpenContractDocExport,
    OpenContractsAnnotationPythonType,
    OpenContractsRelationshipPythonType,
    OpenContractsSinglePageAnnotationType,
    PawlsPagePythonType,
    PawlsTokenPythonType,
    PawlsPageInfo
)
from app.utils.files import check_if_pdf_needs_ocr
from app.utils.layout import reassign_annotation_hierarchy

# Import necessary Docling components based on the new documentation
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import ConversionResult

logger = logging.getLogger(__name__)
# Configure basic logging if running standalone, Uvicorn might override this
# logging.basicConfig(level=logging.INFO)

# --- Global variable for pre-loaded converter (optional, for optimization) ---
# Keep this pattern if desired, but initialization needs adjustment
_global_doc_converter: Optional[DocumentConverter] = None

def _initialize_converter(models_path: str) -> DocumentConverter:
    """Initializes the DocumentConverter with the specified models path."""
    logger.info(f"Initializing DocumentConverter with artifacts_path: {models_path}")
    try:
        # --- NEW INITIALIZATION METHOD ---
        # Create pipeline options specifying the artifacts path
        pipeline_options = PdfPipelineOptions(artifacts_path=models_path)

        # Configure the DocumentConverter with these options for PDF format
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        # --- END NEW INITIALIZATION METHOD ---

        logger.info("DocumentConverter initialized successfully.")
        return doc_converter
    except Exception as e:
        logger.exception(f"Failed to initialize DocumentConverter: {e}")
        raise RuntimeError(f"Could not initialize DocumentConverter. Check models path ('{models_path}') and dependencies.") from e

def get_global_converter(models_path: str) -> DocumentConverter:
    """Gets or initializes the global DocumentConverter instance."""
    global _global_doc_converter
    if _global_doc_converter is None:
        logger.info("Global DocumentConverter not found, initializing...")
        _global_doc_converter = _initialize_converter(models_path)
    else:
        # Optional: Add check if models_path has changed, though unlikely in prod
        logger.debug("Using pre-initialized global DocumentConverter.")
    return _global_doc_converter

# --- Helper Functions ---

def build_text_lookup(docling_document: DoclingDocument) -> Dict[str, str]:
    """
    Creates a lookup dictionary mapping stripped text content to its Docling self_ref ID.

    This is used to find the ID of a heading based on its text content later.

    Args:
        docling_document: The processed DoclingDocument object.

    Returns:
        A dictionary where keys are stripped text strings and values are their self_ref IDs.
    """
    text_lookup: Dict[str, str] = {}
    for text_item in docling_document.texts:
        item_text: Optional[str] = getattr(text_item, "text", None)
        item_ref: Optional[str] = getattr(text_item, "self_ref", None)
        if (
            isinstance(item_text, str)
            and item_text.strip()
            and isinstance(item_ref, str)
        ):
            # Use the stripped text as the key for reliable lookup
            text_lookup[item_text.strip()] = item_ref
    return text_lookup


def convert_docling_item_to_annotation(
    item: Union[TextItem, SectionHeaderItem, ListItem],
    spatial_indices_by_page: Dict[int, STRtree],
    tokens_by_page: Dict[int, List[PawlsTokenPythonType]],
    token_indices_by_page: Dict[int, np.ndarray],
    page_dimensions: Dict[int, Tuple[float, float]],
) -> Optional[OpenContractsAnnotationPythonType]:
    """
    Converts a Docling item (TextItem, SectionHeaderItem, ListItem) into an
    OpenContracts annotation format, linking it to underlying PAWLS tokens.

    It uses spatial indexing (STRtree) to find tokens that intersect with the
    item's bounding box.

    Args:
        item: The Docling item to convert.
        spatial_indices_by_page: A dictionary mapping 0-based page indices to STRtree spatial indices of tokens.
        tokens_by_page: A dictionary mapping 0-based page indices to lists of PawlsTokenPythonType.
        token_indices_by_page: A dictionary mapping 0-based page indices to numpy arrays of original token indices.
        page_dimensions: A dictionary mapping 0-based page indices to (width, height) tuples.

    Returns:
        An OpenContractsAnnotationPythonType object or None if conversion is not possible
        (e.g., missing provenance, bounding box, or page data).
    """
    if not (hasattr(item, "prov") and item.prov):
        logger.warning(f"Item {getattr(item, 'self_ref', 'UNKNOWN')} lacks provenance, skipping annotation.")
        return None

    # Ensure prov is not empty and first element has bbox
    if not item.prov or not item.prov[0].bbox:
        logger.warning(f"Item {getattr(item, 'self_ref', 'UNKNOWN')} provenance lacks bbox, skipping annotation.")
        return None

    first_prov = item.prov[0]
    bbox = first_prov.bbox
    # Docling uses 1-based page numbers, convert to 0-based for OpenContracts
    page_no = first_prov.page_no - 1
    item_text = getattr(item, "text", "")
    item_ref = getattr(item, "self_ref", None)
    if item_ref is None:
        logger.warning(f"Item text='{item_text[:50]}...' lacks self_ref, skipping annotation.")
        return None # Cannot create annotation without an ID

    # --- Get the label safely ---
    item_label_raw = getattr(item, "label", None)
    item_label_str: str
    if isinstance(item_label_raw, DocItemLabel):
        # Assuming DocItemLabel enum values are strings or have a sensible str representation
        item_label_str = str(item_label_raw.value) if hasattr(item_label_raw, 'value') else str(item_label_raw)
    elif isinstance(item_label_raw, str):
        item_label_str = item_label_raw # Use if it's already a string
    else:
        item_label_str = "" # Default to empty string if missing or not an expected type
        if item_label_raw is not None:
             logger.warning(f"Item {item_ref} has unexpected label type: {type(item_label_raw)}. Using empty string.")
    # --- End safe label handling ---


    # Get page height for coordinate transformation
    page_dims = page_dimensions.get(page_no)
    if page_dims is None:
        logger.warning(f"No page dimensions found for page {page_no} (0-based) for item {item_ref}, skipping annotation.")
        return None
    _, page_height = page_dims


    # Transform Y coordinates
    try:
        screen_bottom = float(page_height) - float(bbox.b)
        screen_top = float(page_height) - float(bbox.t)
        left = float(bbox.l)
        right = float(bbox.r)
    except (TypeError, ValueError) as e:
         logger.warning(f"Invalid bbox coordinates for item {item_ref} on page {page_no}: {bbox}. Error: {e}. Skipping annotation.")
         return None


    # Spatial query
    chunk_bbox = box(left, screen_top, right, screen_bottom)
    spatial_index = spatial_indices_by_page.get(page_no)
    tokens = tokens_by_page.get(page_no)
    token_indices_array = token_indices_by_page.get(page_no)

    if spatial_index is None or tokens is None or token_indices_array is None:
        logger.warning(
            f"No spatial index or tokens found for page {page_no} (0-based) for item {item_ref}; skipping annotation."
        )
        return None

    # Perform spatial query safely
    token_ids = [] # Default to empty list
    try:
        candidate_indices = spatial_index.query(chunk_bbox)
        # Ensure candidate_indices is iterable and contains integers if not empty
        if isinstance(candidate_indices, np.ndarray) and candidate_indices.size > 0:
            # Ensure indices are within bounds of the geometries array
            valid_indices = candidate_indices[candidate_indices < len(spatial_index.geometries)]
            if len(valid_indices) < len(candidate_indices):
                 logger.warning(f"Some candidate token indices out of bounds for item {item_ref} on page {page_no}.")

            if len(valid_indices) > 0:
                candidate_geometries = spatial_index.geometries.take(valid_indices)
                # Check for intersection
                intersects_mask = [geom.intersects(chunk_bbox) if geom.is_valid else False for geom in candidate_geometries]
                actual_indices = valid_indices[intersects_mask]

                # Ensure indices are valid for token_indices_array
                if actual_indices.size > 0:
                     valid_actual_indices = actual_indices[actual_indices < len(token_indices_array)]
                     if len(valid_actual_indices) < len(actual_indices):
                          logger.warning(f"Some actual token indices out of bounds for token_indices_array on item {item_ref}, page {page_no}.")

                     if valid_actual_indices.size > 0:
                          token_indices = token_indices_array[valid_actual_indices]
                          token_ids = [
                              {"pageIndex": page_no, "tokenIndex": int(idx)} for idx in sorted(token_indices)
                          ]
                     else:
                          logger.warning(f"No valid actual token indices after bounds check for item {item_ref} on page {page_no}.")
                else:
                     logger.warning(f"No actual tokens intersect bbox for item {item_ref} on page {page_no}.")
            else:
                 logger.warning(f"No valid candidate tokens after bounds check for item {item_ref} on page {page_no}.")
        elif candidate_indices is not None and len(candidate_indices) > 0:
             logger.warning(f"Spatial query for item {item_ref} returned unexpected type: {type(candidate_indices)}")
        # else: No candidate tokens found, token_ids remains empty

    except Exception as e:
        logger.error(f"Error during spatial query for item {item_ref} on page {page_no}: {e}", exc_info=True)
        token_ids = [] # Default to empty if query fails


    # Create the annotation_json structure
    internal_annotation_details: dict[int, OpenContractsSinglePageAnnotationType] = {
        page_no: {
            "bounds": {
                "left": left,
                "top": screen_top,
                "right": right,
                "bottom": screen_bottom,
            },
            "tokensJsons": token_ids,
            "rawText": item_text,
        }
    }

    # Create the full annotation structure
    annotation: OpenContractsAnnotationPythonType = {
        "id": item_ref,
        "annotationLabel": item_label_str,
        "rawText": item_text,
        "page": page_no,
        "annotationJson": internal_annotation_details,
        "parent_id": None,  # Will be assigned later during chunk processing
        "annotation_type": "TOKEN_LABEL",
        "structural": True, # Assuming these are structural elements from Docling
    }

    return annotation


def _extract_title(doc: DoclingDocument, default_title: str) -> str:
    """
    Extracts a title from the DoclingDocument, preferring TITLE or PAGE_HEADER labels.

    Args:
        doc: The processed DoclingDocument.
        default_title: The title to use if none is found in the document.

    Returns:
        The extracted title or the default title.
    """
    for text_item in doc.texts:
        # Check if the label indicates a title or header
        if text_item.label in [DocItemLabel.TITLE, DocItemLabel.PAGE_HEADER]:
            title_text = getattr(text_item, "text", "").strip()
            if title_text: # Ensure the text is not empty
                logger.info(f"Title found (label: {text_item.label}): '{title_text}'")
                return title_text
    logger.info(f"No title found in document items, using default title: '{default_title}'")
    return default_title

def _extract_description(doc: DoclingDocument, title: str) -> str:
    """
    Creates a short description, usually the title plus the first paragraph.

    Args:
        doc: The processed DoclingDocument.
        title: The title extracted from the document.

    Returns:
        A description string.
    """
    description = title
    for text_item in doc.texts:
        # Find the first paragraph to append to the description
        if text_item.label == DocItemLabel.PARAGRAPH:
            paragraph_text = getattr(text_item, "text", "").strip()
            if paragraph_text: # Ensure paragraph is not empty
                description += f"\n{paragraph_text}"
                logger.debug("Appended first paragraph to description.")
                break # Only take the first paragraph
    return description

def _generate_pawls_content(
    doc: DoclingDocument, doc_bytes: bytes, force_ocr: bool = False
) -> Tuple[
    List[PawlsPagePythonType],
    Dict[int, STRtree],
    Dict[int, List[PawlsTokenPythonType]],
    Dict[int, np.ndarray],
    Dict[int, Tuple[float, float]],
    str,
]:
    """
    Generates PAWLS formatted token data and associated spatial indices from a PDF.

    It first tries to extract text using pdfplumber. If that fails, or if `force_ocr`
    is True, it falls back to using OCR (pdf2image + pytesseract).

    Args:
        doc: The DoclingDocument (used mainly for page dimensions).
        doc_bytes: The raw bytes of the PDF file.
        force_ocr: If True, skip pdfplumber and directly use OCR.

    Returns:
        A tuple containing:
        - List of PawlsPagePythonType objects.
        - Dictionary mapping 0-based page index to token STRtree.
        - Dictionary mapping 0-based page index to list of PawlsTokenPythonType.
        - Dictionary mapping 0-based page index to numpy array of original token indices (0..N-1).
        - Dictionary mapping 0-based page index to (width, height) tuple.
        - The full extracted text content concatenated across pages.

    Raises:
        RuntimeError: If essential external dependencies (like Tesseract or Poppler) are missing.
        Exception: If critical errors occur during PDF processing.
    """
    logger.info("Generating PAWLS content...")

    pawls_pages_data: List[PawlsPagePythonType] = []
    spatial_indices_by_page: Dict[int, STRtree] = {}
    tokens_by_page_data: Dict[int, List[PawlsTokenPythonType]] = {}
    token_indices_by_page: Dict[int, np.ndarray] = {}
    page_dimensions: Dict[int, Tuple[float, float]] = {}
    content_parts: List[str] = [] # Store text content per page

    pdf_file_stream = io.BytesIO(doc_bytes)
    needs_ocr = force_ocr or check_if_pdf_needs_ocr(pdf_file_stream)
    pdf_file_stream.seek(0) # Reset stream position after check

    logger.info(f"Initial OCR check: {'Forced' if force_ocr else 'Needed' if needs_ocr else 'Not Needed'}")

    # --- Path 1: Try pdfplumber first (if not force_ocr and not needs_ocr) ---
    pdfplumber_success = False
    if not force_ocr and not needs_ocr:
        logger.info("Attempting text extraction with pdfplumber...")
        import pdfplumber # Keep import here to avoid dependency if only OCR is used

        try:
            with pdfplumber.open(pdf_file_stream) as pdf:
                if not pdf.pages:
                     logger.warning("pdfplumber found no pages in the PDF.")
                     needs_ocr = True # Fallback to OCR if no pages found

                for page_num_1based, pdf_page in enumerate(pdf.pages, start=1):
                    page_num_0based = page_num_1based - 1
                    logger.debug(f"Processing page {page_num_1based} (0-based: {page_num_0based}) with pdfplumber")

                    # Get page dimensions - prefer Docling's calculation if available
                    docling_page: Optional[DoclingPage] = doc.pages.get(page_num_1based)
                    if docling_page and docling_page.size:
                        width = docling_page.size.width
                        height = docling_page.size.height
                        logger.debug(f"Using Docling page size for page {page_num_1based}: ({width}, {height})")
                    else:
                        # Fallback to pdfplumber's dimensions
                        width = float(pdf_page.width)
                        height = float(pdf_page.height)
                        logger.warning(f"Using pdfplumber page size for page {page_num_1based}: ({width}, {height})")

                    page_dimensions[page_num_0based] = (width, height)

                    # Calculate scaling factors if Docling/pdfplumber dimensions differ (unlikely if both succeed)
                    scale_x = width / float(pdf_page.width)
                    scale_y = height / float(pdf_page.height)
                    if abs(scale_x - 1.0) > 1e-3 or abs(scale_y - 1.0) > 1e-3:
                         logger.warning(f"Significant scaling factor for page {page_num_1based}: sx={scale_x}, sy={scale_y}")


                    current_page_tokens: List[PawlsTokenPythonType] = []
                    current_page_geometries: List[box] = []
                    current_page_token_indices: List[int] = [] # Store original 0..N-1 index
                    page_content_parts: List[str] = []

                    # Extract words using pdfplumber
                    # Experiment with extraction options: x_tolerance, y_tolerance, keep_blank_chars, use_text_flow
                    words = pdf_page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False, use_text_flow=True)
                    if not words:
                         logger.warning(f"pdfplumber found no words on page {page_num_1based}.")
                         # Add empty entries to maintain page structure, OCR might find content later
                         content_parts.append("")
                         pawls_page = PawlsPagePythonType(
                             page=PawlsPageInfo(width=width, height=height, index=page_num_1based),
                             tokens=[],
                         )
                         pawls_pages_data.append(pawls_page)
                         continue # Move to next page

                    for token_index, word in enumerate(words):
                        # pdfplumber coordinates: x0, top, x1, bottom (origin top-left)
                        x0 = float(word["x0"]) * scale_x
                        top = float(word["top"]) * scale_y
                        x1 = float(word["x1"]) * scale_x
                        bottom = float(word["bottom"]) * scale_y
                        text = word["text"]

                        # PAWLS uses top-left corner (x, y) and width, height
                        token_x = x0
                        token_y = top
                        token_width = x1 - x0
                        token_height = bottom - top

                        # Skip potentially invalid boxes from pdfplumber
                        if token_width <= 0 or token_height <= 0 or not text.strip():
                            # logger.debug(f"Skipping invalid geometry or empty word on page {page_num_1based}: {word}")
                            continue

                        token = PawlsTokenPythonType(
                            x=token_x,
                            y=token_y,
                            width=token_width,
                            height=token_height,
                            text=text,
                        )
                        current_page_tokens.append(token)
                        page_content_parts.append(text)

                        # Geometry for STRtree (using PAWLS/screen coordinates: minx, miny, maxx, maxy)
                        token_bbox = box(token_x, token_y, token_x + token_width, token_y + token_height)
                        current_page_geometries.append(token_bbox)
                        current_page_token_indices.append(token_index) # Store the original index (0..N-1) for this page

                    content_parts.append(" ".join(page_content_parts))

                    # Build spatial index for the page if tokens were found
                    if current_page_geometries:
                         # STRtree expects an array of geometries
                         geometries_array = np.array(current_page_geometries)
                         token_indices_array = np.array(current_page_token_indices, dtype=np.intp) # Use integer type suitable for indexing
                         spatial_index = STRtree(geometries_array)
                         spatial_indices_by_page[page_num_0based] = spatial_index
                         tokens_by_page_data[page_num_0based] = current_page_tokens
                         token_indices_by_page[page_num_0based] = token_indices_array
                         logger.debug(f"Built STRtree for page {page_num_1based} with {len(current_page_geometries)} tokens (pdfplumber).")
                    else:
                         logger.warning(f"No valid geometries found for page {page_num_1based} using pdfplumber.")
                         # Ensure empty lists/arrays are stored if needed downstream
                         tokens_by_page_data[page_num_0based] = []
                         token_indices_by_page[page_num_0based] = np.array([], dtype=np.intp)


                    # Create PAWLS page structure
                    pawls_page = PawlsPagePythonType(
                        page=PawlsPageInfo(width=width, height=height, index=page_num_1based),
                        tokens=current_page_tokens,
                    )
                    pawls_pages_data.append(pawls_page)

                # If loop completed without major errors, mark pdfplumber as successful
                pdfplumber_success = True
                logger.info("Successfully processed PDF with pdfplumber.")

        except ImportError:
             logger.warning("pdfplumber not installed. Falling back to OCR.")
             needs_ocr = True
        except Exception as e:
             logger.error(f"Error processing PDF with pdfplumber: {e}\n{traceback.format_exc()}")
             logger.warning("Falling back to OCR due to pdfplumber error.")
             needs_ocr = True # Force OCR path now
             # Clear any partial data from failed pdfplumber attempt
             pawls_pages_data.clear()
             spatial_indices_by_page.clear()
             tokens_by_page_data.clear()
             token_indices_by_page.clear()
             page_dimensions.clear()
             content_parts.clear()
             pdf_file_stream.seek(0) # Reset stream

    # --- Path 2: Use OCR (if forced, needed, or pdfplumber failed) ---
    if not pdfplumber_success:
        logger.info("Using OCR (pdf2image + pytesseract) for text extraction.")
        try:
            # Convert PDF pages to images
            try:
                 images = pdf2image.convert_from_bytes(doc_bytes, dpi=300) # Use adequate DPI
            except pdf2image.exceptions.PDFInfoNotInstalledError as e:
                 logger.error("pdfinfo command not found (part of poppler-utils). Please install poppler-utils.")
                 raise RuntimeError("Missing dependency: poppler-utils") from e
            except Exception as img_conv_error:
                 logger.error(f"Failed to convert PDF to images: {img_conv_error}", exc_info=True)
                 raise # Re-raise critical error

            if not images:
                 logger.error("pdf2image conversion resulted in no images.")
                 raise RuntimeError("PDF to image conversion failed.")

            for page_num_1based, page_image in enumerate(images, start=1):
                page_num_0based = page_num_1based - 1
                logger.debug(f"Processing page {page_num_1based} (0-based: {page_num_0based}) via OCR")

                # Get page dimensions - prefer Docling's calculation if available
                docling_page: Optional[DoclingPage] = doc.pages.get(page_num_1based)
                if docling_page and docling_page.size:
                    width = docling_page.size.width
                    height = docling_page.size.height
                    logger.debug(f"Using Docling page size for OCR page {page_num_1based}: ({width}, {height})")
                else:
                    # Fallback to image dimensions (should match if DPI is consistent)
                    img_width_px, img_height_px = page_image.size
                    # Assuming standard PDF points (72 DPI) for width/height if Docling failed
                    # This might need adjustment based on how Docling calculates size
                    width = float(img_width_px) * 72.0 / 300.0 # Example conversion assuming 300 DPI scan
                    height = float(img_height_px) * 72.0 / 300.0
                    logger.warning(f"Using image size ({img_width_px}x{img_height_px}px @300 DPI) converted to points for page {page_num_1based}: ({width:.2f}, {height:.2f})")

                # Store dimensions if not already set by a failed pdfplumber attempt
                if page_num_0based not in page_dimensions:
                     page_dimensions[page_num_0based] = (width, height)
                else:
                     # If pdfplumber failed but set dimensions, keep them? Or update? Let's update.
                     logger.debug(f"Updating page dimensions for page {page_num_0based} based on OCR/image.")
                     page_dimensions[page_num_0based] = (width, height)


                # Perform OCR using pytesseract
                custom_config = r"--oem 3 --psm 3" # Example config: LSTM engine, auto page segmentation
                try:
                    # Use image_to_data to get bounding boxes, text, and confidence
                    ocr_data = pytesseract.image_to_data(
                        page_image,
                        output_type=pytesseract.Output.DICT,
                        config=custom_config,
                        # lang='eng' # Specify language if needed
                    )
                except pytesseract.TesseractNotFoundError:
                     logger.error("Tesseract executable not found. Please install Tesseract and ensure it's in the PATH.")
                     raise RuntimeError("Missing dependency: tesseract-ocr")
                except Exception as ocr_error:
                     logger.error(f"Pytesseract failed for page {page_num_1based}: {ocr_error}", exc_info=True)
                     # Decide how to handle OCR failure for a page: skip page, return error?
                     # Let's skip this page's tokens but add empty content part
                     if len(content_parts) <= page_num_0based: content_parts.append("")
                     if len(pawls_pages_data) <= page_num_0based:
                          pawls_pages_data.append(PawlsPagePythonType(
                              page=PawlsPageInfo(width=width, height=height, index=page_num_1based), tokens=[]
                          ))
                     continue # Skip to next page

                current_page_tokens: List[PawlsTokenPythonType] = []
                current_page_geometries: List[box] = []
                current_page_token_indices: List[int] = [] # Store original 0..N-1 index from OCR data
                page_content_parts: List[str] = []

                img_width_px, img_height_px = page_image.size
                # Scale OCR pixel coordinates (top-left origin) to PDF points (top-left origin for PAWLS)
                scale_x = width / img_width_px
                scale_y = height / img_height_px

                n_boxes = len(ocr_data["text"])
                token_count_on_page = 0
                for i in range(n_boxes):
                    # Tesseract's data includes levels (page, block, para, line, word)
                    # We are interested in word level (level 5)
                    if int(ocr_data["level"][i]) == 5:
                        conf = int(ocr_data["conf"][i])
                        word_text = ocr_data["text"][i]

                        # Filter based on confidence and non-empty text
                        # Confidence is string in dict, convert to int. -1 means not applicable (e.g. whitespace block)
                        if conf > 30 and word_text.strip(): # Use a confidence threshold (e.g., 30)
                            # OCR coordinates are pixel values (top-left origin)
                            x_px = float(ocr_data["left"][i])
                            y_px = float(ocr_data["top"][i])
                            w_px = float(ocr_data["width"][i])
                            h_px = float(ocr_data["height"][i])

                            # Scale to PDF points (top-left origin for PAWLS)
                            token_x = x_px * scale_x
                            token_y = y_px * scale_y
                            token_width = w_px * scale_x
                            token_height = h_px * scale_y

                            if token_width <= 0 or token_height <= 0: continue # Skip invalid boxes

                            token = PawlsTokenPythonType(
                                x=token_x, y=token_y, width=token_width, height=token_height, text=word_text
                            )
                            current_page_tokens.append(token)
                            page_content_parts.append(word_text)

                            # Geometry for STRtree (minx, miny, maxx, maxy)
                            token_bbox = box(token_x, token_y, token_x + token_width, token_y + token_height)
                            current_page_geometries.append(token_bbox)
                            # Store the index relative to the start of *this page's* tokens
                            current_page_token_indices.append(token_count_on_page)
                            token_count_on_page += 1


                # Store content, ensuring list is long enough
                if len(content_parts) <= page_num_0based:
                     content_parts.append(" ".join(page_content_parts))
                elif not content_parts[page_num_0based]: # If pdfplumber failed and left empty string
                     content_parts[page_num_0based] = " ".join(page_content_parts)
                # Else: pdfplumber might have found *some* text, OCR might find more. Overwrite or append? Let's overwrite for simplicity.
                else:
                     logger.debug(f"Overwriting pdfplumber content with OCR content for page {page_num_1based}")
                     content_parts[page_num_0based] = " ".join(page_content_parts)


                # Build spatial index for the page if tokens were found
                if current_page_geometries:
                     geometries_array = np.array(current_page_geometries)
                     token_indices_array = np.array(current_page_token_indices, dtype=np.intp)
                     spatial_index = STRtree(geometries_array)
                     spatial_indices_by_page[page_num_0based] = spatial_index
                     tokens_by_page_data[page_num_0based] = current_page_tokens
                     token_indices_by_page[page_num_0based] = token_indices_array
                     logger.debug(f"Built STRtree for page {page_num_1based} with {len(current_page_geometries)} tokens (OCR).")
                else:
                     logger.warning(f"No valid geometries found via OCR for page {page_num_1based}")
                     tokens_by_page_data[page_num_0based] = []
                     token_indices_by_page[page_num_0based] = np.array([], dtype=np.intp)


                # Create or update PAWLS page data
                if len(pawls_pages_data) <= page_num_0based:
                    pawls_page = PawlsPagePythonType(
                        page=PawlsPageInfo(width=width, height=height, index=page_num_1based),
                        tokens=current_page_tokens,
                    )
                    pawls_pages_data.append(pawls_page)
                else: # Update existing page created by failed pdfplumber attempt
                     pawls_pages_data[page_num_0based].tokens = current_page_tokens
                     pawls_pages_data[page_num_0based].page.width = width
                     pawls_pages_data[page_num_0based].page.height = height
                     logger.debug(f"Updated PAWLS data for page {page_num_1based} with OCR results.")


        except RuntimeError as e: # Catch missing dependencies
             raise e
        except Exception as e:
             logger.error(f"Fatal error during OCR processing: {e}\n{traceback.format_exc()}")
             # Depending on requirements, might want to return partial results or raise error
             raise RuntimeError(f"OCR Processing failed: {e}") from e

    # --- Final Steps ---
    # Concatenate content from all pages
    full_content = "\n".join(content_parts) # Use newline as page separator? Or space?

    # Sanity check lengths
    num_pages = len(page_dimensions)
    if not (len(pawls_pages_data) == num_pages and len(tokens_by_page_data) == num_pages and len(spatial_indices_by_page) == num_pages):
         logger.warning(f"Inconsistent number of pages across generated data structures: "
                        f"Dims={len(page_dimensions)}, PAWLS={len(pawls_pages_data)}, "
                        f"Tokens={len(tokens_by_page_data)}, Indices={len(spatial_indices_by_page)}")
         # This might indicate an issue needing investigation

    logger.info(f"PAWLS content generation complete. Processed {num_pages} pages.")
    return (
        pawls_pages_data,
        spatial_indices_by_page,
        tokens_by_page_data,
        token_indices_by_page,
        page_dimensions,
        full_content,
    )

# --- Original process_document function (slightly modified) ---
# Rename or keep as is, but ensure it doesn't rely on the global 'doc_converter'

def _internal_process_document(
    doc_converter: DocumentConverter, # Accept converter as argument
    pdf_bytes: bytes,
    pdf_filename: str,
    force_ocr: bool = False,
    roll_up_groups: bool = False,
    llm_enhanced_hierarchy: bool = False,
) -> Optional[OpenContractDocExport]:
    """
    Internal processing logic using a provided DocumentConverter instance.

    (Keep the entire logic from the original process_document function here,
     but remove any direct reference to the global 'doc_converter'. It now
     uses the one passed as an argument.)

    Args:
        doc_converter: An initialized DocumentConverter instance.
        pdf_bytes: The content of the PDF file.
        pdf_filename: Original filename.
        force_ocr: If True, forces OCR.
        roll_up_groups: If True, groups items under headings.
        llm_enhanced_hierarchy: If True, attempts LLM hierarchy (Placeholder).

    Returns:
        An OpenContractDocExport object or None if processing fails.
    """
    logger.info(f"Processing document: {pdf_filename} using provided converter")
    logger.info(f"Options: force_ocr={force_ocr}, roll_up_groups={roll_up_groups}, llm_enhanced_hierarchy={llm_enhanced_hierarchy}")

    try:
        buf = BytesIO(pdf_bytes)
        doc_stream = DocumentStream(name=pdf_filename, stream=buf)

        # Convert file via Docling using the provided converter
        result = doc_converter.convert(doc_stream)
        if result.status != ConversionStatus.SUCCESS or not result.document:
            logger.error(f"Docling conversion failed for {pdf_filename}. Status: {result.status}. Errors: {result.errors}")
            # Consider returning specific error info if needed by the client
            return None

        doc: DoclingDocument = cast(DoclingDocument, result.document)
        logger.info(f"Docling conversion successful for {pdf_filename}. Found {len(doc.pages)} pages, {len(doc.texts)} text items.")

        # --- 2. Generate PAWLS Content & Spatial Indices ---
        # This step uses either pdfplumber or OCR based on checks/options
        (
            pawls_pages,
            spatial_indices_by_page,
            tokens_by_page,
            token_indices_by_page, # Map STRtree index -> original token index
            page_dimensions,
            content, # Full text content from extraction
        ) = _generate_pawls_content(doc, pdf_bytes, force_ocr)

        if not pawls_pages:
             logger.error(f"PAWLS content generation failed to produce any pages for {pdf_filename}.")
             return None # Cannot proceed without token/page data

        # --- 3. Run Hierarchical Chunker ---
        logger.debug(f"Running HierarchicalChunker on Docling document...")
        # Ensure HierarchicalChunker is thread-safe if used globally or instantiate here
        chunker = HierarchicalChunker()
        # chunk() returns a generator, convert to list
        chunks = list(chunker.chunk(dl_doc=doc))
        logger.info(f"Generated {len(chunks)} chunks using HierarchicalChunker.")

        # --- 4. Build Base Annotations ---
        logger.debug(f"Converting Docling items to base annotations...")
        base_annotation_lookup: Dict[str, OpenContractsAnnotationPythonType] = {}
        # Build lookup for finding heading refs later
        text_to_ref_lookup = build_text_lookup(doc)

        valid_annotations_count = 0
        conversion_failures = 0
        for item in doc.texts + doc.list_items: # Process both text and list items
            # Ensure item has necessary attributes before passing
            if hasattr(item, "prov") and hasattr(item, "label") and hasattr(item, "self_ref"):
                annotation = convert_docling_item_to_annotation(
                    item, # type: ignore # Pass TextItem or ListItem
                    spatial_indices_by_page,
                    tokens_by_page,
                    token_indices_by_page,
                    page_dimensions,
                )
                if annotation and annotation.id: # Ensure annotation and its ID are valid
                    # Check for duplicate IDs, although self_ref should be unique
                    if annotation.id in base_annotation_lookup:
                         logger.warning(f"Duplicate annotation ID found: {annotation.id}. Overwriting.")
                    base_annotation_lookup[annotation.id] = annotation
                    valid_annotations_count += 1
                elif getattr(item, 'self_ref', None):
                     # Log if conversion failed for an item that had an ID
                     logger.warning(f"Could not convert item with ref {item.self_ref} to annotation.")
                     conversion_failures += 1
            else:
                 logger.debug(f"Skipping item due to missing attributes (prov, label, or self_ref): {type(item)}")
                 conversion_failures += 1


        logger.info(f"Generated {valid_annotations_count} base annotations ({conversion_failures} conversion failures).")
        if not base_annotation_lookup:
             logger.error(f"No base annotations could be generated for {pdf_filename}. Cannot proceed.")
             return None

        # --- 5. Apply Hierarchy and Collect Relationships ---
        logger.debug(f"Applying hierarchy based on chunks (roll_up_groups={roll_up_groups})...")
        # Structure for rolled-up groups: {heading_id: [child_id1, child_id2,...]}
        flattened_heading_annot_id_to_children: Dict[str, List[str]] = {}
        # Structure for non-rolled-up groups: [(heading_id1, [child_id1,...]), (heading_id1, [child_id5,...]), ...]
        # Allows multiple relationship groups per heading if chunker splits them
        heading_annot_id_to_chunk_children: List[Tuple[str, List[str]]] = []

        processed_item_refs = set() # Keep track of items assigned to a chunk/parent

        for i, chunk in enumerate(chunks):
            parent_ref: Optional[str] = None
            heading_text: Optional[str] = None

            # Identify the parent heading for this chunk
            if chunk.meta.headings:
                # Assuming the first heading is the primary one for the chunk
                heading_text = chunk.meta.headings[0].strip()
                parent_ref = text_to_ref_lookup.get(heading_text)

                if not parent_ref:
                     logger.warning(f"Chunk {i}: Could not find self_ref for heading text: '{heading_text}'. Chunk items will have no parent.")
                # Ensure the identified heading exists as a valid annotation
                elif parent_ref not in base_annotation_lookup:
                     logger.warning(f"Chunk {i}: Heading text '{heading_text}' with ref '{parent_ref}' not found in base annotations. Cannot use as parent.")
                     parent_ref = None # Treat as parentless chunk

            # Prepare structures for relationship building
            if parent_ref:
                if roll_up_groups:
                    # Ensure entry exists for this heading
                    if parent_ref not in flattened_heading_annot_id_to_children:
                        flattened_heading_annot_id_to_children[parent_ref] = []
                else:
                    # For non-rollup, add a new tuple representing this specific chunk's relationship to the parent
                    # We will populate the child list in the next step
                    heading_annot_id_to_chunk_children.append((parent_ref, []))


            # Process items within the chunk and assign parent_id
            current_chunk_children: List[str] = []
            if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                for item in chunk.meta.doc_items:
                    item_ref = getattr(item, 'self_ref', None)
                    if not item_ref: continue

                    annotation = base_annotation_lookup.get(item_ref)
                    if annotation:
                        # Assign parent_id if a valid parent_ref exists for this chunk
                        # and the item is not the heading itself
                        if parent_ref and parent_ref != annotation.id:
                            # Check if already assigned a parent (can happen if item appears in multiple chunks?)
                            if annotation.parent_id and annotation.parent_id != parent_ref:
                                 logger.warning(f"Annotation {annotation.id} ('{annotation.rawText[:20]}...') found in chunk {i} under parent {parent_ref}, but already has parent {annotation.parent_id}. Overwriting parent.")
                            annotation.parent_id = parent_ref
                            current_chunk_children.append(annotation.id)
                            processed_item_refs.add(annotation.id)
                        # else: # Item has no parent for this chunk (either no heading or item is the heading)
                            # logger.debug(f"Item {annotation.id} in chunk {i} has no parent assigned.")
                            # pass # Keep parent_id as None

                    # else: # Annotation might be missing if conversion failed earlier
                       # logger.debug(f"Annotation not found for item ref {item_ref} in chunk {i}")

            # Add the collected children to the appropriate relationship structure
            if parent_ref and current_chunk_children:
                if roll_up_groups:
                     flattened_heading_annot_id_to_children[parent_ref].extend(current_chunk_children)
                else:
                     # Add children to the *last* tuple added for this parent_ref
                     # This assumes the list `heading_annot_id_to_chunk_children` maintains chunk order
                     if heading_annot_id_to_chunk_children and heading_annot_id_to_chunk_children[-1][0] == parent_ref:
                          heading_annot_id_to_chunk_children[-1][1].extend(current_chunk_children)
                     else:
                          # This case should ideally not happen if a tuple was added above when parent_ref was found
                          logger.error(f"Logic error: Could not find matching tuple for parent_ref {parent_ref} in non-rollup list for chunk {i}.")


        # --- 6. Build Relationships ---
        logger.debug("Building relationship objects...")
        relationships: List[OpenContractsRelationshipPythonType] = []
        rel_counter = 0
        if roll_up_groups:
            for heading_id, child_ids in flattened_heading_annot_id_to_children.items():
                # Only create relationship if there are children and the heading exists
                if child_ids and heading_id in base_annotation_lookup:
                    relationships.append(OpenContractsRelationshipPythonType(
                        id=f"group-rel-{rel_counter}",
                        relationshipLabel="Docling Group Relationship", # Rolled-up relationship
                        source_annotation_ids=[heading_id],
                        target_annotation_ids=child_ids,
                        structural=True,
                    ))
                    rel_counter += 1
                elif not child_ids:
                     logger.debug(f"Skipping relationship for heading {heading_id}: No children found in roll-up.")
                elif heading_id not in base_annotation_lookup:
                     logger.warning(f"Skipping relationship for heading {heading_id}: Heading annotation not found.")

        else: # Non-rolled-up relationships (one per chunk under a heading)
            for heading_id, child_ids in heading_annot_id_to_chunk_children:
                 # Only create relationship if there are children and the heading exists
                if child_ids and heading_id in base_annotation_lookup:
                    relationships.append(OpenContractsRelationshipPythonType(
                        id=f"group-rel-{rel_counter}",
                        relationshipLabel="Docling Chunk Relationship", # Relationship per chunk
                        source_annotation_ids=[heading_id],
                        target_annotation_ids=child_ids,
                        structural=True,
                    ))
                    rel_counter += 1
                elif not child_ids:
                     logger.debug(f"Skipping relationship for heading {heading_id} (chunk instance): No children found.")
                elif heading_id not in base_annotation_lookup:
                     logger.warning(f"Skipping relationship for heading {heading_id} (chunk instance): Heading annotation not found.")


        logger.info(f"Generated {len(relationships)} relationships.")

        # --- 7. Final Output Assembly ---
        # Get all annotations from the lookup
        final_annotations = list(base_annotation_lookup.values())

        # Optional LLM Enhancement Step
        if llm_enhanced_hierarchy:
            logger.info("Applying LLM-enhanced hierarchy (Placeholder)...")
            # This function currently just returns the input
            final_annotations = reassign_annotation_hierarchy(final_annotations)

        # Extract metadata
        default_title = Path(pdf_filename).stem # Use filename stem as default
        title = _extract_title(doc, default_title)
        description = _extract_description(doc, title)

        # Construct the final export object using Pydantic model
        open_contracts_data = OpenContractDocExport(
            title=title,
            content=content, # Use the text extracted during PAWLS generation
            description=description,
            pawlsFileContent=pawls_pages, # Use alias
            pageCount=len(pawls_pages), # Use alias
            docLabels=[], # Add logic if doc labels are needed/extracted
            labelledText=final_annotations, # Use alias
            relationships=relationships,
        )

        logger.info(f"Successfully processed {pdf_filename} via internal function.")
        return open_contracts_data

    except RuntimeError as dep_error:
         logger.error(f"Processing failed for {pdf_filename} (internal): {dep_error}", exc_info=True)
         return None
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error(f"Unexpected error during internal processing for {pdf_filename}: {e}\n{stacktrace}")
        return None


# --- New Wrapper Function for Dynamic Initialization ---

def process_document_dynamic_init(
    pdf_bytes: bytes,
    pdf_filename: str,
    models_path: str,
    force_ocr: bool = False,
    roll_up_groups: bool = False,
    llm_enhanced_hierarchy: bool = False,
) -> Optional[OpenContractDocExport]:
    """
    Processes a PDF document using a dynamically initialized DocumentConverter,
    extracts metadata, annotations, and relationships.

    Args:
        pdf_bytes: The raw bytes of the PDF file.
        pdf_filename: The original filename (for logging/metadata).
        models_path: Path to the directory containing downloaded Docling models.
        force_ocr: Flag to force OCR even if text layer exists.
        roll_up_groups: Flag to enable roll-up groups feature for relationships.
        llm_enhanced_hierarchy: Flag for LLM enhanced hierarchy processing.

    Returns:
        An OpenContractDocExport object if successful, None otherwise.
    """
    logger.info(f"Starting dynamic processing for {pdf_filename}...")
    logger.info(f"Options: force_ocr={force_ocr}, roll_up_groups={roll_up_groups}, llm_enhanced_hierarchy={llm_enhanced_hierarchy}")

    # --- 1. Initialize DocumentConverter (Dynamically) ---
    try:
        if not os.path.exists(models_path):
            logger.error(f"Docling models path '{models_path}' does not exist.")
            return None

        # Configure Docling options
        ocr_options = EasyOcrOptions(
            model_storage_directory=models_path,
            # Add other OCR options if needed, e.g., gpu=True if supported/desired
        )
        pipeline_options = PdfPipelineOptions(
            artifacts_path=models_path,
            do_ocr=True, # Let Docling decide based on its internal logic unless overridden? Or rely on our force_ocr?
                       # Setting True ensures OCR models are loaded if needed.
                       # The _generate_pawls_content function will ultimately decide based on check_if_pdf_needs_ocr or force_ocr.
            do_table_structure=True, # Keep table structure extraction enabled
            generate_page_images=True, # Needed for OCR path in _generate_pawls_content
            ocr_options=ocr_options,
            # Add other pipeline options if needed
        )
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        logger.info("DocumentConverter initialized dynamically.")
    except Exception as e:
        logger.error(f"Failed to initialize DocumentConverter: {e}", exc_info=True)
        return None

    # --- 2. Convert PDF using Docling ---
    doc_stream = DocumentStream(name=pdf_filename, stream=BytesIO(pdf_bytes))
    try:
        conv_res = doc_converter.convert(doc_stream)
        if conv_res.status != ConversionStatus.SUCCESS or not conv_res.document:
            logger.error(f"Docling conversion failed for {pdf_filename}. Status: {conv_res.status}, Errors: {conv_res.errors}")
            return None
        doc: DoclingDocument = cast(DoclingDocument, conv_res.document)
        logger.info(f"Successfully converted {pdf_filename} using Docling.")

    except Exception as e:
        logger.error(f"Exception during Docling conversion for {pdf_filename}: {e}", exc_info=True)
        return None

    # --- 3. Generate PAWLS Content and Base Data ---
    try:
        (
            pawls_pages,
            spatial_indices_by_page,
            tokens_by_page,
            token_indices_by_page,
            page_dimensions,
            content,
        ) = _generate_pawls_content(doc, pdf_bytes, force_ocr)

        if not pawls_pages:
             logger.error(f"Failed to generate PAWLS content for {pdf_filename}.")
             return None

    except Exception as e:
        logger.error(f"Exception during PAWLS generation for {pdf_filename}: {e}", exc_info=True)
        return None

    # --- 4. Extract Metadata ---
    default_title = Path(pdf_filename).stem
    extracted_title = _extract_title(doc, default_title)
    extracted_description = _extract_description(doc, extracted_title)
    logger.info(f"Extracted Title: '{extracted_title}'")
    # logger.debug(f"Extracted Description: '{extracted_description}'") # Can be long

    # --- 5. Build Base Annotations and Text Lookup ---
    base_annotation_lookup: dict[str, OpenContractsAnnotationPythonType] = {}
    text_lookup: dict[str, str] = {}
    try:
        text_lookup = build_text_lookup(doc)
        # logger.debug(f"Built text lookup with {len(text_lookup)} entries.")

        valid_annotations_count = 0
        for text_item in doc.texts:
            annotation = convert_docling_item_to_annotation(
                text_item,
                spatial_indices_by_page,
                tokens_by_page,
                token_indices_by_page,
                page_dimensions,
            )
            if annotation and annotation.get("id"):
                base_annotation_lookup[annotation["id"]] = annotation
                valid_annotations_count += 1
            elif annotation:
                 logger.warning(f"Generated annotation missing ID for text: {getattr(text_item, 'text', '')[:50]}...")
            # else: Annotation failed, warning already logged in convert_ function

        logger.info(f"Generated {valid_annotations_count} base annotations from {len(doc.texts)} text items.")
        if not base_annotation_lookup:
             logger.warning(f"No base annotations could be generated for {pdf_filename}. Proceeding without annotations/relationships.")
             # Decide if this should be an error or continue with empty lists

    except Exception as e:
        logger.error(f"Exception during base annotation generation for {pdf_filename}: {e}", exc_info=True)
        return None # Treat failure here as critical

    # --- 6. Chunk Document and Build Hierarchy/Relationships ---
    final_annotations: list[OpenContractsAnnotationPythonType] = []
    relationships: list[OpenContractsRelationshipPythonType] = []
    try:
        chunker = HierarchicalChunker()
        chunks = list(chunker.chunk(dl_doc=doc))
        logger.info(f"Generated {len(chunks)} chunks using HierarchicalChunker.")

        # Data structures for relationship building
        flattened_heading_annot_id_to_children: dict[str, list[str]] = {}
        heading_annot_id_to_children: list[tuple[Optional[str], list[str]]] = [] # Allow None for parent_ref

        processed_annotation_ids = set()

        for i, chunk in enumerate(chunks):
            parent_ref: Optional[str] = None
            if chunk.meta.headings:
                if len(chunk.meta.headings) > 1:
                    logger.warning(f"Chunk {i} has multiple headings ({len(chunk.meta.headings)}); using the first one.")
                heading_text = chunk.meta.headings[0].strip()
                parent_ref = text_lookup.get(heading_text)
                if not parent_ref:
                    logger.warning(f"Could not find annotation ref for heading text: '{heading_text}' in chunk {i}.")
                # else: logger.debug(f"Chunk {i} heading '{heading_text}' maps to ref: {parent_ref}")

                # Initialize parent in relationship structures if needed
                if parent_ref:
                    if roll_up_groups:
                        if parent_ref not in flattened_heading_annot_id_to_children:
                            flattened_heading_annot_id_to_children[parent_ref] = []
                    # For non-rolled-up, we add tuples later as children are found

            # Process items within the chunk
            current_chunk_children: list[str] = []
            if hasattr(chunk.meta, "doc_items"):
                # logger.debug(f"Chunk {i} has {len(chunk.meta.doc_items)} doc_items.")
                for item in chunk.meta.doc_items:
                    item_ref = getattr(item, "self_ref", None)
                    if not item_ref:
                        logger.warning(f"Doc item in chunk {i} lacks self_ref, cannot link.")
                        continue

                    annotation = base_annotation_lookup.get(item_ref)
                    if annotation:
                        # Assign parent_id to the annotation IN THE LOOKUP
                        annotation["parent_id"] = parent_ref
                        processed_annotation_ids.add(item_ref) # Mark as processed

                        # Add to relationship structures
                        if parent_ref:
                            current_chunk_children.append(item_ref)
                            if roll_up_groups:
                                # Check ensures parent_ref exists from heading check above
                                flattened_heading_annot_id_to_children[parent_ref].append(item_ref)
                        # else: Item has no parent heading in this chunk

                    else:
                        # This can happen if convert_docling_item_to_annotation failed for this item earlier
                        logger.warning(f"Annotation not found in base_annotation_lookup for item ref '{item_ref}' in chunk {i}.")

            # Add non-rolled-up relationship entry for this chunk's parent/children
            if not roll_up_groups and current_chunk_children:
                 # We add even if parent_ref is None, representing top-level items for this chunk
                 heading_annot_id_to_children.append((parent_ref, current_chunk_children))


        # Add any annotations that weren't part of any chunk's doc_items
        # These might be headers themselves or items missed by chunking logic
        unprocessed_annotations = []
        for ref, annot in base_annotation_lookup.items():
            if ref not in processed_annotation_ids:
                 # Keep existing parent_id if somehow assigned, otherwise it remains None
                 unprocessed_annotations.append(annot)
                 processed_annotation_ids.add(ref) # Add here to avoid duplicates if logic changes

        # Combine processed annotations (which now have parent_ids) and unprocessed ones
        final_annotations = list(base_annotation_lookup.values())
        logger.info(f"Total final annotations: {len(final_annotations)} ({len(unprocessed_annotations)} were not in chunk doc_items).")


        # Build relationship objects
        rel_counter = 0
        if roll_up_groups:
            logger.info(f"Building rolled-up relationships from {len(flattened_heading_annot_id_to_children)} headings.")
            for heading_id, child_ids in flattened_heading_annot_id_to_children.items():
                 # Ensure heading_id itself exists as an annotation
                 if heading_id not in base_annotation_lookup:
                      logger.warning(f"Heading ID '{heading_id}' for rolled-up relationship not found in annotations. Skipping relationship.")
                      continue
                 # Ensure child_ids exist as annotations
                 valid_child_ids = [cid for cid in child_ids if cid in base_annotation_lookup]
                 if len(valid_child_ids) < len(child_ids):
                      logger.warning(f"Found {len(child_ids) - len(valid_child_ids)} missing child annotations for heading '{heading_id}'.")

                 if not valid_child_ids:
                      logger.warning(f"No valid child annotations found for heading '{heading_id}'. Skipping relationship.")
                      continue

                 relationship_entry: OpenContractsRelationshipPythonType = {
                    "id": f"group-rel-{rel_counter}",
                    "relationshipLabel": "GROUP", # Use standard label?
                    "source_annotation_ids": [heading_id],
                    "target_annotation_ids": valid_child_ids,
                    "structural": True,
                 }
                 relationships.append(relationship_entry)
                 rel_counter += 1
        else:
            logger.info(f"Building non-rolled-up relationships from {len(heading_annot_id_to_children)} chunk groups.")
            for heading_id, child_ids in heading_annot_id_to_children:
                 # Allow relationships for top-level items (heading_id is None) ?
                 # Current spec requires source_annotation_ids. If None, maybe skip?
                 if heading_id is None:
                      logger.debug(f"Skipping relationship for chunk group with no heading ID (found {len(child_ids)} children).")
                      continue

                 if heading_id not in base_annotation_lookup:
                      logger.warning(f"Heading ID '{heading_id}' for non-rolled-up relationship not found in annotations. Skipping relationship.")
                      continue

                 valid_child_ids = [cid for cid in child_ids if cid in base_annotation_lookup]
                 if len(valid_child_ids) < len(child_ids):
                      logger.warning(f"Found {len(child_ids) - len(valid_child_ids)} missing child annotations for non-rolled-up heading '{heading_id}'.")

                 if not valid_child_ids:
                      logger.warning(f"No valid child annotations found for non-rolled-up heading '{heading_id}'. Skipping relationship.")
                      continue

                 relationship_entry: OpenContractsRelationshipPythonType = {
                    "id": f"group-rel-{rel_counter}",
                    "relationshipLabel": "GROUP", # Use standard label?
                    "source_annotation_ids": [heading_id],
                    "target_annotation_ids": valid_child_ids,
                    "structural": True,
                 }
                 relationships.append(relationship_entry)
                 rel_counter += 1

        logger.info(f"Generated {len(relationships)} relationships.")

    except Exception as e:
        logger.error(f"Exception during chunking or relationship building for {pdf_filename}: {e}", exc_info=True)
        # Decide whether to return partial results or fail
        # Returning partial results might be okay if annotations are generated but relationships fail
        # For now, let's return None on failure in this critical step.
        return None


    # --- 7. Optional: LLM Enhanced Hierarchy ---
    if llm_enhanced_hierarchy:
        logger.info("Applying LLM-enhanced hierarchy (experimental)...")
        try:
            # Ensure the function handles the current annotation format
            # Note: This function might modify parent_ids, potentially conflicting
            # with the relationships just built. Careful coordination is needed.
            # It might be better to run this *before* building relationships
            # if it primarily adjusts parent_ids.
            # Or, the relationship building needs to re-run based on new parent_ids.
            # For now, applying it after relationship building based on original code placement.
            enriched_annotations = reassign_annotation_hierarchy(final_annotations)
            if enriched_annotations:
                 final_annotations = enriched_annotations
                 logger.info("LLM-enhanced hierarchy applied.")
            else:
                 logger.warning("LLM-enhanced hierarchy function returned None or empty list.")
        except Exception as e:
            logger.error(f"Exception during LLM-enhanced hierarchy processing: {e}", exc_info=True)
            # Continue with non-enriched data if LLM step fails

    # --- 8. Construct Final Export Data ---
    export_data: OpenContractDocExport = {
        "title": extracted_title,
        "content": content,
        "description": extracted_description,
        "pageCount": len(pawls_pages),
        "pawlsFileContent": pawls_pages,
        "docLabels": [], # Placeholder - Add logic if document-level labels are needed
        "labelledText": final_annotations,
        "relationships": relationships,
    }

    logger.info(f"Successfully processed {pdf_filename}. Returning OpenContractDocExport.")
    return OpenContractDocExport(**export_data)

# ... rest of the file (DoclingParser class, etc.) ...

# Make sure helper functions like _generate_pawls_content, _extract_title,
# _extract_description, convert_docling_item_to_annotation, build_text_lookup
# are defined within or imported into this file scope.
# Also ensure reassign_annotation_hierarchy is available if llm_enhanced_hierarchy is used.
