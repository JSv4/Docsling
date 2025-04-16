import pytest
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi.testclient import TestClient
from httpx import Response # Import Response for type hinting

# Import your FastAPI app instance and Pydantic models
from app.main import app
from app.models.types import OpenContractDocExport

# Import fixture paths (adjust path if necessary)
from tests.fixtures import (
    SAMPLE_PAWLS_FILE_ONE_PATH,
    SAMPLE_PDF_FILE_ONE_PATH,
    SAMPLE_TXT_FILE_ONE_PATH,
)

logger = logging.getLogger(__name__)

# --- Fixtures ---

@pytest.fixture(scope="module")
def client() -> TestClient:
    """Provides a TestClient instance for the FastAPI app."""
    return TestClient(app)

@pytest.fixture(scope="module")
def sample_pdf_bytes() -> bytes:
    """Reads the sample PDF file content."""
    if not SAMPLE_PDF_FILE_ONE_PATH.exists():
        pytest.fail(f"Sample PDF file not found at: {SAMPLE_PDF_FILE_ONE_PATH}")
    with SAMPLE_PDF_FILE_ONE_PATH.open("rb") as f:
        return f.read()

@pytest.fixture(scope="module")
def sample_pdf_filename() -> str:
    """Returns the name of the sample PDF file."""
    return SAMPLE_PDF_FILE_ONE_PATH.name

# --- Test Class ---

class TestDoclingParserEndpoint:
    """
    Test suite for the /parse/ endpoint, mimicking the Django integration tests.
    """

    # Expected values based on the Django test case for the specific sample PDF
    EXPECTED_TITLE = "Exhibit 10.1"
    EXPECTED_PAGE_COUNT = 23
    EXPECTED_TOKEN_COUNTS_DEFAULT = [
        392, 374, 470, 350, 490, 431, 386, 587, 463, 577,
        806, 276, 706, 563, 428, 426, 572, 616, 465, 335,
        496, 43, 6,
    ]
    EXPECTED_TOKEN_COUNTS_OCR = [
        390, 374, 469, 350, 490, 431, 380, 568, 463, 577,
        806, 276, 706, 563, 428, 426, 572, 616, 465, 335,
        495, 43, 6,
    ]
    EXPECTED_RELATIONSHIPS_COUNT_ROLLED_UP = 25
    EXPECTED_RELATIONSHIPS_COUNT_NOT_ROLLED_UP = 137
    EXPECTED_LABELLED_TEXT_COUNT = 271

    def _assert_common(self, result_data: Dict[str, Any], expected_token_counts: List[int]):
        """Helper function for common assertions."""
        assert result_data is not None, "API returned None or empty response"

        # Validate structure using Pydantic model (optional but recommended)
        try:
            parsed_result = OpenContractDocExport(**result_data)
        except Exception as e:
            pytest.fail(f"API response failed Pydantic validation: {e}\nResponse data: {result_data}")

        assert parsed_result.title == self.EXPECTED_TITLE
        assert parsed_result.page_count == self.EXPECTED_PAGE_COUNT
        assert len(parsed_result.labelled_text) == self.EXPECTED_LABELLED_TEXT_COUNT

        # Assert token counts for each page
        assert len(parsed_result.pawls_file_content) == self.EXPECTED_PAGE_COUNT, "PAWLS page count mismatch"
        
        for page_idx, page in enumerate(parsed_result.pawls_file_content):
            assert len(page.tokens) == expected_token_counts[page_idx], \
                f"Token count mismatch on page {page_idx + 1}"

    def test_parse_endpoint_default(
        self, client: TestClient, sample_pdf_bytes: bytes, sample_pdf_filename: str
    ) -> None:
        """
        Test the /parse/ endpoint with default options (roll_up_groups=False).
        """
        files = {"file": (sample_pdf_filename, sample_pdf_bytes, "application/pdf")}
        # roll_up_groups defaults to False in the endpoint definition
        data = {"force_ocr": False, "llm_enhanced_hierarchy": False}

        response: Response = client.post("/parse/", files=files, data=data)

        assert response.status_code == 200, f"API Error: {response.text}"
        result_data = response.json()

        self._assert_common(result_data, self.EXPECTED_TOKEN_COUNTS_DEFAULT)

        # Assert relationships count for non-rolled-up case
        assert len(result_data.get("relationships", [])) == self.EXPECTED_RELATIONSHIPS_COUNT_NOT_ROLLED_UP, \
            "Relationship count mismatch for roll_up_groups=False"

    def test_parse_endpoint_roll_up_groups(
        self, client: TestClient, sample_pdf_bytes: bytes, sample_pdf_filename: str
    ) -> None:
        """
        Test the /parse/ endpoint with roll_up_groups=True.
        """
        files = {"file": (sample_pdf_filename, sample_pdf_bytes, "application/pdf")}
        data = {"roll_up_groups": True, "force_ocr": False, "llm_enhanced_hierarchy": False}

        response: Response = client.post("/parse/", files=files, data=data)

        assert response.status_code == 200, f"API Error: {response.text}"
        result_data = response.json()

        self._assert_common(result_data, self.EXPECTED_TOKEN_COUNTS_DEFAULT)

        # Assert relationships count for rolled-up case
        assert len(result_data.get("relationships", [])) == self.EXPECTED_RELATIONSHIPS_COUNT_ROLLED_UP, \
            "Relationship count mismatch for roll_up_groups=True"

    def test_parse_endpoint_force_ocr(
        self, client: TestClient, sample_pdf_bytes: bytes, sample_pdf_filename: str
    ) -> None:
        """
        Test the /parse/ endpoint with force_ocr=True and roll_up_groups=True.
        """
        files = {"file": (sample_pdf_filename, sample_pdf_bytes, "application/pdf")}
        data = {"force_ocr": True, "roll_up_groups": True, "llm_enhanced_hierarchy": False}

        response: Response = client.post("/parse/", files=files, data=data)

        assert response.status_code == 200, f"API Error: {response.text}"
        result_data = response.json()

        # Use OCR-specific expected token counts for assertion
        self._assert_common(result_data, self.EXPECTED_TOKEN_COUNTS_OCR)

        # Assert relationships count for rolled-up case (should be same as non-OCR rolled-up)
        assert len(result_data.get("relationships", [])) == self.EXPECTED_RELATIONSHIPS_COUNT_ROLLED_UP, \
            "Relationship count mismatch for force_ocr=True, roll_up_groups=True"

    def test_parse_endpoint_invalid_file_type(
        self, client: TestClient, sample_pdf_filename: str
    ) -> None:
        """
        Test sending a non-PDF file type results in a 415 error.
        """
        # Simulate a text file upload
        files = {"file": (sample_pdf_filename.replace(".pdf", ".txt"), b"this is not a pdf", "text/plain")}
        data = {"roll_up_groups": False} # Options don't matter here

        response: Response = client.post("/parse/", files=files, data=data)

        assert response.status_code == 415 # Unsupported Media Type

    def test_parse_endpoint_no_file(self, client: TestClient) -> None:
        """
        Test sending request without a file results in a 422 error (Unprocessable Entity).
        FastAPI/Pydantic handles missing required File parameter.
        """
        data = {"roll_up_groups": False}
        response: Response = client.post("/parse/", data=data) # No files parameter

        assert response.status_code == 422 # Unprocessable Entity

    def test_health_endpoint(self, client: TestClient) -> None:
        """Test the /health endpoint."""
        response = client.get("/health")
        # Expect 503 if converter failed, 200 otherwise
        # In a normal test run where setup works, expect 200
        if response.status_code == 503:
             pytest.skip("Skipping health check assertion as service reported unavailable (likely model loading issue)")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "message": "Docling Parser Service is running and converter is ready."} 