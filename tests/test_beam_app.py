import pytest
import base64
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Import the function decorated with @beam.endpoint
from beam_app import parse_pdf_beam

# Import fixture paths (adjust if necessary)
from tests.fixtures import SAMPLE_PDF_FILE_ONE_PATH

# --- Fixtures ---

@pytest.fixture(scope="module")
def sample_pdf_bytes_beam() -> bytes:
    """Reads the sample PDF file content for Beam tests."""
    if not SAMPLE_PDF_FILE_ONE_PATH.exists():
        pytest.fail(f"Sample PDF file not found at: {SAMPLE_PDF_FILE_ONE_PATH}")
    with SAMPLE_PDF_FILE_ONE_PATH.open("rb") as f:
        return f.read()

@pytest.fixture(scope="module")
def sample_pdf_base64_beam(sample_pdf_bytes_beam: bytes) -> str:
    """Provides the sample PDF content as a base64 encoded string."""
    return base64.b64encode(sample_pdf_bytes_beam).decode('utf-8')

@pytest.fixture(scope="module")
def sample_pdf_filename_beam() -> str:
    """Returns the name of the sample PDF file."""
    return SAMPLE_PDF_FILE_ONE_PATH.name

# --- Test Class for Beam Endpoint Function ---

# Mock the models path check if needed for local testing without actual models
# This depends on whether your tests run where the models are accessible
@pytest.fixture(autouse=True)
def mock_models_path(monkeypatch):
    """
    Mocks os.path.exists/isdir for the expected models path if models
    aren't available in the local test environment. Adjust path as needed.
    """
    expected_path = "/app/docling_models" # Path used inside beam_app.py
    def mock_exists(path):
        if path == expected_path:
            return True
        return os.path.exists(path) # Call original for other paths

    def mock_isdir(path):
        if path == expected_path:
            return True
        return os.path.isdir(path) # Call original for other paths

    # Only mock if the path doesn't actually exist locally relative to test run
    # This check might need adjustment based on your test execution context
    if not Path("./docling_models").exists():
         print(f"\nMocking model path checks for: {expected_path}")
         monkeypatch.setattr(os.path, "exists", mock_exists)
         monkeypatch.setattr(os.path, "isdir", mock_isdir)
    else:
         print(f"\nUsing actual model path: ./docling_models (assuming relative path works)")
         # If models *are* present, you might need to adjust the path
         # used inside beam_app.py for local testing or ensure the test
         # runner finds them correctly relative to the expected '/app/docling_models'.
         # For simplicity, the mock assumes they *aren't* easily accessible
         # relative to the '/app/' path during local tests.


class TestBeamEndpointFunction:
    """
    Tests the parse_pdf_beam function directly, simulating Beam inputs.
    """

    # Reuse expected values from TestDoclingParserEndpoint if applicable
    EXPECTED_TITLE = "Exhibit 10.1"
    EXPECTED_PAGE_COUNT = 23
    EXPECTED_RELATIONSHIPS_COUNT_ROLLED_UP = 24
    EXPECTED_LABELLED_TEXT_COUNT = 272

    def test_beam_function_success_rolled_up(
        self, sample_pdf_base64_beam: str, sample_pdf_filename_beam: str
    ) -> None:
        """
        Test the beam function directly with valid inputs and roll_up_groups=True.
        """
        # Simulate the input dictionary Beam would provide
        test_inputs = {
            "pdf_base64": sample_pdf_base64_beam,
            "filename": sample_pdf_filename_beam,
            "roll_up_groups": True,
            "force_ocr": False,
        }

        # Call the function directly
        response_dict = parse_pdf_beam(**test_inputs)

        # Assertions on the returned dictionary
        assert "error" not in response_dict, f"Function returned error: {response_dict.get('error')}"
        assert "result" in response_dict
        result = response_dict["result"]

        assert result is not None
        assert result.get("title") == self.EXPECTED_TITLE
        assert result.get("pageCount") == self.EXPECTED_PAGE_COUNT # Check alias used in model dump
        assert len(result.get("labelledText", [])) == self.EXPECTED_LABELLED_TEXT_COUNT
        assert len(result.get("relationships", [])) == self.EXPECTED_RELATIONSHIPS_COUNT_ROLLED_UP
        # Add more specific checks on pawlsFileContent, etc. if needed

    def test_beam_function_missing_input(self) -> None:
        """Test calling the function with missing required input."""
        test_inputs = {
            "filename": "test.pdf"
            # Missing pdf_base64
        }
        response_dict = parse_pdf_beam(**test_inputs)
        assert "error" in response_dict
        assert "pdf_base64" in response_dict["error"]

    def test_beam_function_invalid_base64(self) -> None:
        """Test calling the function with invalid base64."""
        test_inputs = {
            "pdf_base64": "this is not valid base64",
            "filename": "test.pdf"
        }
        response_dict = parse_pdf_beam(**test_inputs)
        assert "error" in response_dict
        assert "Invalid base64" in response_dict["error"]

    # Add more tests for other options (force_ocr, roll_up_groups=False)
    # and potential error conditions within the parsing logic itself. 