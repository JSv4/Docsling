import pytest
import base64
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, patch

# Import the mock from conftest (it's already applied to sys.modules)
from conftest import beam_mock

# Import the function decorated with @beam.endpoint
import beam
from beam_app import parse_pdf_beam

# Import fixture paths (adjust if necessary)
from tests.fixtures import SAMPLE_PDF_FILE_ONE_PATH
from app.models.types import OpenContractDocExport

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

@pytest.fixture(scope="function")
def mock_context() -> beam.Context:
    """Creates a mock beam.Context object."""
    mock_ctx = MagicMock(spec=beam.Context)
    mock_ctx.on_start_value = None
    return mock_ctx

@pytest.fixture(scope="function")
def mock_doc_converter() -> MagicMock:
    """Creates a mock DocumentConverter object."""
    return MagicMock(name="MockDocConverter")

@pytest.fixture(scope="function")
def sample_success_response() -> OpenContractDocExport:
    """Provides a sample successful parsing result model."""
    return OpenContractDocExport(
        title="Sample Title",
        content="Sample content...",
        pageCount=1,
        pawlsFileContent=[],
        docLabels=[],
        labelledText=[],
        relationships=[]
    )

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

# Test data
@pytest.fixture
def sample_beam_input():
    return {
        "filename": "test.pdf",
        "pdf_base64": base64.b64encode(b"%PDF-1.0 test").decode('utf-8'),
        "force_ocr": False,
        "roll_up_groups": True
    }

@pytest.fixture
def mock_context():
    context = MagicMock()
    # Mock the on_start_value to simulate a loaded DocumentConverter
    context.on_start_value = MagicMock()
    return context

# Tests
def test_parse_pdf_beam_validates_inputs(mock_context):
    # Test with missing required inputs
    result = parse_pdf_beam(mock_context, filename="test.pdf")
    assert "error" in result
    assert "pdf_base64" in result["error"]
    
    result = parse_pdf_beam(mock_context, pdf_base64="invalid")
    assert "error" in result
    assert "filename" in result["error"]

@patch('beam_app.base64.b64decode')
@patch('beam_app._internal_process_document')
def test_parse_pdf_beam_processes_document(mock_process, mock_b64decode, mock_context, sample_beam_input):
    # Setup mocks
    mock_b64decode.return_value = b"mocked pdf content"
    mock_process.return_value = MagicMock()
    mock_process.return_value.model_dump.return_value = {"title": "Test Doc"}
    
    # Call the function
    result = parse_pdf_beam(mock_context, **sample_beam_input)
    
    # Verify
    assert "result" in result
    assert result["result"] == {"title": "Test Doc"}
    mock_process.assert_called_once()

# Add more tests as needed

class TestBeamAppFunction:
    """Tests for the parse_pdf_beam function."""

    # Reuse expected values from TestDoclingParserEndpoint if applicable
    EXPECTED_TITLE = "Exhibit 10.1"
    EXPECTED_PAGE_COUNT = 23
    EXPECTED_RELATIONSHIPS_COUNT_ROLLED_UP = 24
    EXPECTED_LABELLED_TEXT_COUNT = 272

    def test_beam_function_success(
        self,
        mocker,
        mock_context: beam.Context,
        mock_doc_converter: MagicMock,
        sample_pdf_base64_beam: str,
        sample_pdf_filename_beam: str,
        sample_success_response: OpenContractDocExport,
    ) -> None:
        """Test successful execution of the beam function."""
        # --- Arrange ---
        mock_context.on_start_value = mock_doc_converter
        mock_process = mocker.patch(
            "app.core.parser._internal_process_document",
            return_value=sample_success_response
        )
        test_inputs = {
            "pdf_base64": sample_pdf_base64_beam,
            "filename": sample_pdf_filename_beam,
            "roll_up_groups": True,
            "force_ocr": False,
        }

        # --- Act ---
        response_dict = parse_pdf_beam(mock_context, **test_inputs)

        # --- Assert ---
        mock_process.assert_called_once_with(
            doc_converter=mock_doc_converter,
            pdf_bytes=mocker.ANY,
            pdf_filename=sample_pdf_filename_beam,
            force_ocr=False,
            roll_up_groups=True,
            llm_enhanced_hierarchy=False,
        )
        assert "result" in response_dict
        assert "error" not in response_dict
        result = response_dict["result"]
        assert result is not None
        assert result == sample_success_response.model_dump(mode='json')

    def test_beam_function_loader_failed(
        self,
        mock_context: beam.Context,
        sample_pdf_base64_beam: str,
        sample_pdf_filename_beam: str,
    ) -> None:
        """Test the case where the on_start loader failed (context value is None)."""
        mock_context.on_start_value = None
        test_inputs = {
            "pdf_base64": sample_pdf_base64_beam,
            "filename": sample_pdf_filename_beam,
        }
        response_dict = parse_pdf_beam(mock_context, **test_inputs)
        assert "error" in response_dict
        assert "result" not in response_dict
        assert "Parser service initialization failed" in response_dict["error"]

    def test_beam_function_processing_error(
        self,
        mocker,
        mock_context: beam.Context,
        mock_doc_converter: MagicMock,
        sample_pdf_base64_beam: str,
        sample_pdf_filename_beam: str,
    ) -> None:
        """Test the case where _internal_process_document raises an exception."""
        mock_context.on_start_value = mock_doc_converter
        mock_process = mocker.patch(
            "app.core.parser._internal_process_document",
            side_effect=ValueError("Something went wrong during processing")
        )
        test_inputs = {
            "pdf_base64": sample_pdf_base64_beam,
            "filename": sample_pdf_filename_beam,
        }
        response_dict = parse_pdf_beam(mock_context, **test_inputs)
        mock_process.assert_called_once()
        assert "error" in response_dict
        assert "result" not in response_dict
        assert "An unexpected internal server error occurred during processing" in response_dict["error"]
        assert "ValueError" in response_dict["error"]

    def test_beam_function_processing_returns_none(
        self,
        mocker,
        mock_context: beam.Context,
        mock_doc_converter: MagicMock,
        sample_pdf_base64_beam: str,
        sample_pdf_filename_beam: str,
    ) -> None:
        """Test the case where _internal_process_document returns None."""
        mock_context.on_start_value = mock_doc_converter
        mock_process = mocker.patch(
            "app.core.parser._internal_process_document",
            return_value=None
        )
        test_inputs = {
            "pdf_base64": sample_pdf_base64_beam,
            "filename": sample_pdf_filename_beam,
        }
        response_dict = parse_pdf_beam(mock_context, **test_inputs)
        mock_process.assert_called_once()
        assert "error" in response_dict
        assert "result" not in response_dict
        assert "Failed to process document" in response_dict["error"]

    def test_beam_function_missing_input(
        self,
        mock_context: beam.Context,
        mock_doc_converter: MagicMock,
    ) -> None:
        """Test calling the function with missing required input."""
        mock_context.on_start_value = mock_doc_converter
        test_inputs = {
            "filename": "test.pdf"
        }
        response_dict = parse_pdf_beam(mock_context, **test_inputs)
        assert "error" in response_dict
        assert "pdf_base64" in response_dict["error"]

    def test_beam_function_invalid_base64(
        self,
        mock_context: beam.Context,
        mock_doc_converter: MagicMock,
    ) -> None:
        """Test calling the function with invalid base64."""
        mock_context.on_start_value = mock_doc_converter
        test_inputs = {
            "pdf_base64": "this is not valid base64",
            "filename": "test.pdf"
        }
        response_dict = parse_pdf_beam(mock_context, **test_inputs)
        assert "error" in response_dict
        assert "Invalid base64" in response_dict["error"]

    # Add more tests for other options (force_ocr, roll_up_groups=False)
    # by adjusting the inputs and the expected call to the mock_process function. 