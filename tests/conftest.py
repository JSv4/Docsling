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
from app.models.types import OpenContractDocExport # Import the response model

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
        # Define the mock result matching test_main.py expectations
        mock_result = OpenContractDocExport(
            title="Exhibit 10.1",          # Matches EXPECTED_TITLE
            content="Mocked content",      # Content doesn't need to be exact for mock
            description="Mocked description",
            pageCount=23,                  # Matches EXPECTED_PAGE_COUNT
            pawlsFileContent=[],
            docLabels=[],
            labelledText=[],
            relationships=[]
        )

        def mock_process(*args, **kwargs):
            print(f"Mocked process_document_dynamic_init called with args: {args}, kwargs: {kwargs}")
            return mock_result

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