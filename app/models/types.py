import numpy as np
from pydantic import BaseModel, Field
from typing import Optional, Union, List, Dict, Tuple, Any, Literal

# --- Request Models ---

class ParseRequest(BaseModel):
    """Defines the expected JSON body for the /parse/ endpoint."""
    filename: str = Field(..., description="The original filename of the PDF.")
    pdf_base64: str = Field(..., description="Base64 encoded string of the PDF content.")
    force_ocr: bool = Field(False, description="Force OCR processing even if text is detectable.")
    roll_up_groups: bool = Field(False, description="Roll up items under the same heading into single relationships.")
    llm_enhanced_hierarchy: bool = Field(False, description="Apply experimental LLM-based hierarchy enhancement.")

# --- PAWLS Types ---
# Based on PAWLS format (https://github.com/allenai/pawls)

class PawlsTokenPythonType(BaseModel):
    """Represents a single token (word) in PAWLS format."""
    x: float = Field(description="Left x-coordinate of the token bounding box.")
    y: float = Field(description="Top y-coordinate of the token bounding box.")
    width: float = Field(description="Width of the token bounding box.")
    height: float = Field(description="Height of the token bounding box.")
    text: str = Field(description="The text content of the token.")

class PawlsPageInfo(BaseModel):
    """Metadata for a page in PAWLS format."""
    width: float = Field(description="Width of the page.")
    height: float = Field(description="Height of the page.")
    index: int = Field(description="1-based index of the page.") # Docling uses 1-based page indices

class PawlsPagePythonType(BaseModel):
    """Represents a single page containing tokens in PAWLS format."""
    page: PawlsPageInfo
    tokens: List[PawlsTokenPythonType]

# --- OpenContracts Annotation Types ---
# Based on the structure expected by OpenContracts frontend/consumers

class Point(BaseModel):
    """Represents a pointer to a specific token on a specific page."""
    pageIndex: int = Field(description="0-based index of the page.")
    tokenIndex: int = Field(description="0-based index of the token within the page's token list.")

class Bounds(BaseModel):
    """Represents a bounding box in screen coordinates (origin top-left)."""
    left: float
    top: float
    right: float
    bottom: float

class OpenContractsSinglePageAnnotationType(BaseModel):
    """Details of an annotation specific to a single page."""
    bounds: Bounds = Field(description="Bounding box of the annotation on this page.")
    # Field alias used to allow 'tokens_jsons' in Python but serialize/deserialize as 'tokensJsons'
    tokens_jsons: List[Point] = Field(default_factory=list, alias="tokensJsons", description="List of token pointers comprising this annotation fragment.")
    rawText: str = Field(description="The raw text corresponding to this part of the annotation on this page.")

    class Config:
        allow_population_by_field_name = True # Allows using 'tokens_jsons' during model creation

class OpenContractsAnnotationPythonType(BaseModel):
    """Represents a labelled annotation spanning potentially multiple pages."""
    id: Optional[str] = Field(None, description="Unique identifier for the annotation (e.g., Docling self_ref).")
    annotationLabel: str = Field(description="The label assigned to the annotation (e.g., 'Section Header', 'Paragraph').")
    rawText: str = Field(description="The full raw text of the annotation.")
    page: int = Field(description="0-based index of the primary page where the annotation starts or is most relevant.")
    annotation_json: Dict[int, OpenContractsSinglePageAnnotationType] = Field(description="Mapping of 0-based page indices to page-specific annotation details.")
    parent_id: Optional[str] = Field(None, description="ID of the parent annotation in a hierarchy.")
    annotation_type: str = Field("TOKEN_LABEL", description="Type of annotation, defaults to 'TOKEN_LABEL'.")
    structural: bool = Field(True, description="Indicates if the annotation is structural (derived from layout).")

# --- OpenContracts Relationship Types ---

class OpenContractsRelationshipPythonType(BaseModel):
    """Represents a relationship between annotations."""
    id: str = Field(description="Unique identifier for the relationship (e.g., 'group-rel-0').")
    relationshipLabel: str = Field(description="Label describing the relationship type (e.g., 'Docling Group Relationship').")
    source_annotation_ids: List[str] = Field(description="List of IDs of the source annotations.")
    target_annotation_ids: List[str] = Field(description="List of IDs of the target annotations.")
    structural: bool = Field(True, description="Indicates if the relationship is structural.")

# --- OpenContracts Document Export Type ---

class OpenContractDocExport(BaseModel):
    """The final structured data exported for a document."""
    title: str = Field(description="The extracted or default title of the document.")
    content: str = Field(description="Full text content extracted from the document.")
    description: str = Field(description="A short description, often derived from the title and initial paragraphs.")
    # Using alias for consistency
    pawls_file_content: List[PawlsPagePythonType] = Field(alias="pawlsFileContent", description="Document content represented in PAWLS format (pages and tokens).")
    page_count: int = Field(alias="pageCount", description="Total number of pages in the document.")
    doc_labels: List[Any] = Field(default_factory=list, alias="docLabels", description="List of document-level labels (currently unused).")
    labelled_text: List[OpenContractsAnnotationPythonType] = Field(alias="labelledText", description="List of all annotations extracted from the document.")
    relationships: List[OpenContractsRelationshipPythonType] = Field(description="List of relationships identified between annotations.")

    class Config:
        allow_population_by_field_name = True # Allows using aliases during model creation

# --- Helper Types ---
# These might not be strictly necessary for the API contract but useful internally

class BBoxInternal(BaseModel):
    """Internal representation for bounding box calculations if needed."""
    l: float
    t: float
    r: float
    b: float 