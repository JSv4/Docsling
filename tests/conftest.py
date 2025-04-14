import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Create a mock for the beam module before it's imported
beam_mock = MagicMock()
beam_mock.endpoint = lambda **kwargs: lambda func: func  # Make endpoint decorator a no-op
beam_mock.Image.return_value = MagicMock()
beam_mock.Image.return_value.add_commands.return_value = MagicMock()
beam_mock.Context = MagicMock

# Apply the mock to sys.modules so it's used when beam is imported
sys.modules['beam'] = beam_mock

# Now the rest of the imports can proceed
import pytest_asyncio
from httpx import AsyncClient
from typing import AsyncGenerator, Generator

# Make sure the app path is correct relative to the tests directory
# This assumes your tests run from the root 'docling_parser_service' directory
from app.main import app
from app.models.types import OpenContractDocExport # Import the response model

# --- Mocking Setup ---
# You might want to mock external dependencies globally here if needed,
# e.g., the Docling converter initialization if it's problematic during testing.
# from unittest.mock import patch
# patch('app.core.parser.doc_converter', MagicMock()).start()


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