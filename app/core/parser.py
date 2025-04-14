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
    BoundingBox as DoclingBBox, # Explicitly import Docling's BBox
    PageItem as DoclingPage, # Explicitly import Docling's Page
)
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from shapely.geometry import box
from shapely.strtree import STRtree

# Import from our microservice structure
from app.core.config import settings
from app.models.types import (
    OpenContractDocExport,
    OpenContractsAnnotationPythonType,
    OpenContractsRelationshipPythonType,
    OpenContractsSinglePageAnnotationType,
    PawlsPagePythonType,
    PawlsTokenPythonType,
    PawlsPageInfo,
    Point,
    Bounds,
)
from app.utils.files import check_if_pdf_needs_ocr
from app.utils.layout import reassign_annotation_hierarchy

logger = logging.getLogger(__name__)
# Configure basic logging if running standalone, Uvicorn might override this
# logging.basicConfig(level=logging.INFO)


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
        logger.debug(f"Item {getattr(item, 'self_ref', 'UNKNOWN')} has no provenance, cannot convert to annotation.")
        return None

    # Assuming provenance is ordered and the first one is representative for bbox/page
    first_prov = item.prov[0]
    if not first_prov.bbox or not isinstance(first_prov.bbox, DoclingBBox):
        logger.debug(f"Item {getattr(item, 'self_ref', 'UNKNOWN')} provenance lacks a valid bounding box.")
        return None

    # Docling uses 1-based page numbers, convert to 0-based for internal use
    page_no_0_based = first_prov.page_no - 1
    item_text: str = getattr(item, "text", "")
    item_id: Optional[str] = getattr(item, "self_ref", None)
    item_label: DocItemLabel = getattr(item, "label", DocItemLabel.UNKNOWN)

    page_width, page_height = page_dimensions.get(page_no_0_based, (0.0, 0.0))
    if page_height == 0.0 or page_width == 0.0:
        logger.warning(f"No valid page dimensions found for page {page_no_0_based} (item {item_id}). Cannot calculate screen coordinates.")
        return None

    # Convert Docling BBox (origin bottom-left) to screen coordinates (origin top-left)
    # Docling BBox: l, b, r, t (left, bottom, right, top)
    # Screen Bounds: left, top, right, bottom
    docling_bbox: DoclingBBox = first_prov.bbox
    screen_left = docling_bbox.l
    screen_right = docling_bbox.r
    # PDF Y coordinates increase upwards, Screen Y coordinates increase downwards
    screen_top = page_height - docling_bbox.t
    screen_bottom = page_height - docling_bbox.b

    # Ensure coordinates are valid
    if screen_left >= screen_right or screen_top >= screen_bottom:
         logger.warning(f"Invalid screen coordinates calculated for item {item_id} on page {page_no_0_based}: L={screen_left}, T={screen_top}, R={screen_right}, B={screen_bottom}. Skipping token search.")
         # Create annotation without tokens? Or skip entirely? Let's skip token search for now.
         token_ids = []
    else:
        # Create a Shapely box for spatial querying using screen coordinates
        # Note: PAWLS tokens are stored with top-left origin, matching screen coordinates.
        item_screen_bbox = box(screen_left, screen_top, screen_right, screen_bottom)

        # Retrieve spatial index and token data for the page
        spatial_index = spatial_indices_by_page.get(page_no_0_based)
        page_tokens = tokens_by_page.get(page_no_0_based) # List of PawlsTokenPythonType
        page_token_indices_array = token_indices_by_page.get(page_no_0_based) # Array of original indices (0..N-1)

        if spatial_index is None or page_tokens is None or page_token_indices_array is None:
            logger.warning(
                f"Missing spatial index, tokens, or token indices for page {page_no_0_based} (item {item_id}). Cannot find associated tokens."
            )
            token_ids = []
        else:
            try:
                # Query the STRtree for candidate token indices intersecting the item's bbox
                # The STRtree was built using geometries derived from PawlsTokenPythonType coordinates (x, y, x+width, y+height)
                candidate_indices = spatial_index.query(item_screen_bbox)

                # Refine candidates by checking actual intersection (STRtree query might return items in the envelope)
                # Ensure geometries are available and compatible with shapely STRtree usage
                # Note: Accessing `geometries` directly might be deprecated or internal; `query` should return indices.
                # We need the original token indices corresponding to the geometries found.
                intersecting_token_original_indices = []
                geometries = spatial_index.geometries_array # Access the array used to build the tree
                for idx in candidate_indices:
                    # Check intersection between the item bbox and the token geometry at the candidate index
                    if item_screen_bbox.intersects(geometries[idx]):
                        # Map the index within the STRtree's geometry array back to the original token index
                        original_token_index = page_token_indices_array[idx]
                        intersecting_token_original_indices.append(original_token_index)


                # Create Point objects using the 0-based page index and 0-based token index
                token_ids = [
                    Point(pageIndex=page_no_0_based, tokenIndex=int(idx))
                    for idx in sorted(intersecting_token_original_indices)
                ]
                # logger.debug(f"Found {len(token_ids)} tokens for item {item_id} on page {page_no_0_based}")

            except Exception as query_error:
                 logger.error(f"Error querying STRtree for item {item_id} on page {page_no_0_based}: {query_error}", exc_info=True)
                 token_ids = []


    # Create the annotation structure for this specific page
    annotation_json_page = OpenContractsSinglePageAnnotationType(
        bounds=Bounds(
            left=screen_left,
            top=screen_top,
            right=screen_right,
            bottom=screen_bottom,
        ),
        tokensJsons=token_ids, # Use field name expected by frontend
        rawText=item_text, # Text for this specific item/page (might differ if annotation spans pages)
    )

    # Create the main annotation object
    annotation = OpenContractsAnnotationPythonType(
        id=item_id,
        annotationLabel=str(item_label.value), # Use the string value of the enum
        rawText=item_text, # Full text of the item
        page=page_no_0_based, # Primary page index (0-based)
        annotationJson={page_no_0_based: annotation_json_page}, # Use alias
        parent_id=None, # Parent ID will be assigned later based on hierarchy
        annotation_type="TOKEN_LABEL",
        structural=True,
    )

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
            if chunk.meta and chunk.meta.headings:
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
    models_path: str, # Path to models within the execution environment
    force_ocr: bool = False,
    roll_up_groups: bool = False,
    llm_enhanced_hierarchy: bool = False,
) -> Optional[OpenContractDocExport]:
    """
    Processes a PDF, initializing the Docling converter dynamically.

    This is suitable for environments like serverless functions where global
    initialization might be problematic or needs environment-specific paths.

    Args:
        pdf_bytes: The content of the PDF file.
        pdf_filename: Original filename.
        models_path: The filesystem path to the Docling models directory.
        force_ocr: If True, forces OCR.
        roll_up_groups: If True, groups items under headings.
        llm_enhanced_hierarchy: If True, attempts LLM hierarchy (Placeholder).

    Returns:
        An OpenContractDocExport object or None if processing fails.
    """
    try:
        logger.info(f"Dynamically initializing DocumentConverter with models path: {models_path}")
        if not os.path.exists(models_path) or not os.path.isdir(models_path):
             # Log a critical error and return None if models aren't found
             logger.error(f"Docling models path '{models_path}' does not exist or is not a directory.")
             raise FileNotFoundError(f"Docling models path '{models_path}' not found.")

        # Initialize converter inside the function
        ocr_options = EasyOcrOptions(model_storage_directory=models_path)
        pipeline_options = PdfPipelineOptions(
            artifacts_path=models_path,
            do_ocr=True,
            do_table_structure=True,
            generate_page_images=True,
            ocr_options=ocr_options,
        )
        local_doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        logger.info("Dynamic DocumentConverter initialized successfully.")

        # Call the internal processing logic with the initialized converter
        return _internal_process_document(
            doc_converter=local_doc_converter,
            pdf_bytes=pdf_bytes,
            pdf_filename=pdf_filename,
            force_ocr=force_ocr,
            roll_up_groups=roll_up_groups,
            llm_enhanced_hierarchy=llm_enhanced_hierarchy,
        )

    except FileNotFoundError as fnf_error:
        logger.error(f"Failed dynamic initialization: {fnf_error}")
        return None # Return None if models path issue prevents initialization
    except Exception as e:
        logger.error(f"Failed to dynamically initialize Docling converter or process document: {e}", exc_info=True)
        return None


# --- Original process_document (Optional - Keep for FastAPI if needed) ---
# This version relies on the globally initialized 'doc_converter'
# You might rename it or remove it if the FastAPI app also switches
# to dynamic initialization.

# def process_document(
#     pdf_bytes: bytes,
#     pdf_filename: str,
#     force_ocr: bool = False,
#     roll_up_groups: bool = False,
#     llm_enhanced_hierarchy: bool = False,
# ) -> Optional[OpenContractDocExport]:
#     """
#     Processes a PDF byte stream using the globally initialized Docling converter.
#     (This is the original function structure)
#     """
#     if not doc_converter:
#          logger.error("Global Docling converter not initialized. Cannot process document.")
#          # Optionally, try dynamic init as a fallback?
#          # return process_document_dynamic_init(...)
#          return None
#
#     # Call the internal logic using the global converter
#     return _internal_process_document(
#         doc_converter=doc_converter,
#         # ... pass other args ...
#     ) 