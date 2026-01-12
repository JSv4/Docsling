import sys
import os
import pytest

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest_asyncio
from httpx import AsyncClient
from typing import AsyncGenerator, Generator

# Make sure the app path is correct relative to the tests directory
# This assumes your tests run from the root 'docling_parser_service' directory
from app.main import app
from app.models.types import (
    OpenContractDocExport,
    PawlsPagePythonType,
    PawlsPageInfo,
    PawlsTokenPythonType,
    OpenContractsAnnotationPythonType,
    OpenContractsSinglePageAnnotationType,
    OpenContractsRelationshipPythonType,
    Bounds,
)


def _build_mock_result(roll_up_groups: bool = False, force_ocr: bool = False) -> OpenContractDocExport:
    """Build mock result matching test_main.py expectations."""
    # Expected token counts from test_main.py
    if force_ocr:
        expected_token_counts = [
            390, 374, 469, 350, 490, 431, 380, 568, 463, 577,
            806, 276, 706, 563, 428, 426, 572, 616, 465, 335,
            495, 43, 6,
        ]
    else:
        expected_token_counts = [
            392, 374, 470, 350, 490, 431, 386, 587, 463, 577,
            806, 276, 706, 563, 428, 426, 572, 616, 465, 335,
            496, 43, 6,
        ]

    # Generate mock PAWLS pages with correct token counts
    pawls_pages = []
    for page_idx, token_count in enumerate(expected_token_counts):
        tokens = [
            PawlsTokenPythonType(x=0.0, y=0.0, width=10.0, height=10.0, text=f"token{i}")
            for i in range(token_count)
        ]
        page = PawlsPagePythonType(
            page=PawlsPageInfo(width=612.0, height=792.0, index=page_idx + 1),
            tokens=tokens
        )
        pawls_pages.append(page)

    # Generate 271 mock annotations (EXPECTED_LABELLED_TEXT_COUNT)
    labelled_text = []
    for i in range(271):
        annotation = OpenContractsAnnotationPythonType(
            id=f"annot-{i}",
            annotationLabel="paragraph",
            rawText=f"Mock text {i}",
            page=i % 23,
            annotation_json={
                i % 23: OpenContractsSinglePageAnnotationType(
                    bounds=Bounds(left=0.0, top=0.0, right=100.0, bottom=20.0),
                    tokensJsons=[],
                    rawText=f"Mock text {i}"
                )
            },
            parent_id=None,
            annotation_type="TOKEN_LABEL",
            structural=True
        )
        labelled_text.append(annotation)

    # Generate relationships based on roll_up_groups
    if roll_up_groups:
        rel_count = 25  # EXPECTED_RELATIONSHIPS_COUNT_ROLLED_UP
    else:
        rel_count = 137  # EXPECTED_RELATIONSHIPS_COUNT_NOT_ROLLED_UP

    relationships = [
        OpenContractsRelationshipPythonType(
            id=f"group-rel-{i}",
            relationshipLabel="GROUP",
            source_annotation_ids=[f"annot-{i}"],
            target_annotation_ids=[f"annot-{(i+1) % 271}"],
            structural=True
        )
        for i in range(rel_count)
    ]

    return OpenContractDocExport(
        title="Exhibit 10.1",
        content="Mocked content",
        description="Mocked description",
        pageCount=23,
        pawlsFileContent=pawls_pages,
        docLabels=[],
        labelledText=labelled_text,
        relationships=relationships
    )


# --- Conditional Mocking for FastAPI Tests ---
@pytest.fixture(autouse=True)
def maybe_mock_process_document_dynamic_init(monkeypatch):
    """
    Conditionally mocks the process_document_dynamic_init function based on
    the DOCLING_TEST_WITH_REAL_PARSER environment variable.

    If DOCLING_TEST_WITH_REAL_PARSER=1, the real function is used.
    Otherwise (default), the function is mocked to return a predefined result
    that matches the expectations in test_main.py.
    """
    use_real_parser = os.environ.get("DOCLING_TEST_WITH_REAL_PARSER") == "1"

    if use_real_parser:
        print("\nRunning FastAPI tests with REAL parser (DOCLING_TEST_WITH_REAL_PARSER=1)")
        # Ensure models are accessible if running real parser
        models_path = os.environ.get("DOCLING_MODELS_PATH", "./docling_models")
        if not os.path.exists(models_path) or not os.path.isdir(models_path):
             pytest.skip(f"Real parser requested, but models not found at DOCLING_MODELS_PATH='{models_path}'. Skipping relevant tests.")
        # No mocking needed, the real function will be called
        yield # Allow the test to run
    else:
        print("\nRunning FastAPI tests with MOCKED parser (default)")

        def mock_process(*args, **kwargs):
            roll_up = kwargs.get("roll_up_groups", False)
            force_ocr = kwargs.get("force_ocr", False)
            print(f"Mocked process_document_dynamic_init called with roll_up_groups={roll_up}, force_ocr={force_ocr}")
            return _build_mock_result(roll_up_groups=roll_up, force_ocr=force_ocr)

        # Apply the mock
        monkeypatch.setattr("app.main.process_document_dynamic_init", mock_process)
        yield # Allow the test to run
        # Mock is automatically removed after the test by monkeypatch

# --- Fixtures ---

@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """
    Specify the async backend for pytest-asyncio and httpx.
    'asyncio' is standard.
    """
    return "asyncio"

@pytest_asyncio.fixture(scope="function") # Use function scope for clean state per test
async def client() -> AsyncGenerator[AsyncClient, None]:
    """
    Provides an asynchronous test client for the FastAPI application.
    """
    # Use 'async with' for proper startup/shutdown event handling if defined in app
    async with AsyncClient(app=app, base_url="http://test") as async_client:
        yield async_client
    # Cleanup runs after yield

@pytest.fixture(scope="function")
def sample_parse_result() -> OpenContractDocExport:
    """
    Provides a sample valid OpenContractDocExport object for mocking successful parsing.
    """
    # Create a minimal valid instance of the response model
    # Adjust fields as necessary to match expected structure
    return OpenContractDocExport(
        title="Sample Title",
        content="Sample content.",
        description="Sample description.",
        pawlsFileContent=[], # Keep minimal for mock
        pageCount=1,
        docLabels=[],
        labelledText=[], # Keep minimal for mock
        relationships=[] # Keep minimal for mock
    )

@pytest.fixture(scope="function")
def mock_pdf_file() -> Generator[bytes, None, None]:
    """
    Provides simple mock PDF content as bytes.
    Note: This is NOT a valid PDF, just bytes for testing upload mechanics.
    """
    # A minimal byte string. For real PDF structure testing, use actual small PDFs.
    yield b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Count 0>>endobj\nxref\n0 3\n0000000000 65535 f\n0000000010 00000 n\n0000000058 00000 n\ntrailer<</Size 3/Root 1 0 R>>\nstartxref\n108\n%%EOF" 