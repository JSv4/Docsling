import requests # Use requests library
import base64
import os
import argparse
import logging
import json
from typing import Dict, Any
from dotenv import load_dotenv # <-- Import dotenv

# --- Load environment variables from .env file ---
load_dotenv() # <-- Load .env variables

# Configure logging (same as before)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def submit_pdf_to_deployed_beam(
    endpoint_url: str, # <-- Renamed from serve_url
    api_key: str,      # <-- Renamed from bearer_token
    pdf_path: str,
    force_ocr: bool = False,
    roll_up_groups: bool = False,
    llm_enhanced_hierarchy: bool = False,
) -> Dict[str, Any]:
    """
    Submits a PDF file to a specified DEPLOYED Beam endpoint for parsing,
    using Bearer token authentication as per Beam documentation.

    Args:
        endpoint_url: The HTTP URL of the deployed Beam endpoint.
        api_key: The Beam API Key (used as the Bearer token value).
        pdf_path: The local path to the PDF file to submit.
        force_ocr: Optional flag to force OCR processing.
        roll_up_groups: Optional flag to roll up groups.
        llm_enhanced_hierarchy: Optional flag to use LLM enhancement.

    Returns:
        A dictionary containing the response from the Beam endpoint.
    """
    logger.info(f"Preparing to submit PDF via HTTP POST: {pdf_path} to {endpoint_url}")

    # --- 1. Validate PDF Path (same as before) ---
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found at path: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if not pdf_path.lower().endswith(".pdf"):
        logger.error(f"File is not a PDF: {pdf_path}")
        raise ValueError("File must be a PDF.")

    # --- 2. Read and Encode PDF (same as before) ---
    try:
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
        logger.info("PDF read and base64 encoded successfully.")
    except Exception as e:
        logger.error(f"Failed to read or encode PDF: {e}")
        raise

    # --- 3. Construct Payload (same as before) ---
    filename = os.path.basename(pdf_path)
    payload = {
        "pdf_base64": pdf_base64,
        "filename": filename,
        "force_ocr": force_ocr,
        "roll_up_groups": roll_up_groups,
        "llm_enhanced_hierarchy": llm_enhanced_hierarchy,
    }
    logger.info(f"Payload constructed for filename: {filename}")

    # --- 4. Prepare Headers using Bearer Token Auth ---
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}", # <-- REVERT TO BEARER TOKEN FORMAT
    }

    # --- 5. Send HTTP POST Request (same logic, different auth header) ---
    logger.info("Sending HTTP POST request...")
    try:
        # Add a reasonable timeout (e.g., 5 minutes for potentially long parsing)
        response = requests.post(endpoint_url, headers=headers, json=payload, timeout=300)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        logger.info(f"Request successful with status code: {response.status_code}")
        return response.json() # Return the JSON response body

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP request failed: {e}")
        error_content = e.response.text if e.response else "No response content"
        raise RuntimeError(f"HTTP request failed: {e} - Response: {error_content}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response: {e}")
        raise RuntimeError(f"Invalid JSON response received: {response.text}")


if __name__ == "__main__":
    # --- Update description ---
    parser = argparse.ArgumentParser(description="Submit a PDF via HTTP POST to a DEPLOYED Beam endpoint using Bearer token auth.")

    # --- Update arguments to use .env defaults ---
    parser.add_argument(
        "--endpoint-url", # <-- Renamed argument
        type=str,
        default=os.environ.get("BEAM_ENDPOINT"), # <-- Default from .env
        help="The HTTP URL of the deployed Beam endpoint (can also be set via BEAM_ENDPOINT env var in .env).",
    )
    parser.add_argument(
        "--api-key", # <-- Renamed argument
        type=str,
        default=os.environ.get("BEAM_ENDPOINT_TOKEN"), # <-- Default from .env
        help="The Beam API Key for the endpoint, used as Bearer token (can also be set via BEAM_ENDPOINT_TOKEN env var in .env).",
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the local PDF file to submit.",
    )
    # Optional flags remain the same
    parser.add_argument("--force-ocr", action="store_true", help="Force OCR processing.")
    parser.add_argument("--roll-up-groups", action="store_true", help="Enable roll-up groups.")
    parser.add_argument("--llm-enhanced-hierarchy", action="store_true", help="Enable LLM enhanced hierarchy.")

    args = parser.parse_args()

    # --- Validate credentials/URL are provided ---
    if not args.endpoint_url:
        parser.error("Beam Endpoint URL is required. Set --endpoint-url or BEAM_ENDPOINT environment variable in .env.")
    if not args.api_key:
        parser.error("Beam API Key is required. Set --api-key or BEAM_ENDPOINT_TOKEN environment variable in .env.")

    try:
        # --- Call the renamed function with updated args ---
        result = submit_pdf_to_deployed_beam(
            endpoint_url=args.endpoint_url,
            api_key=args.api_key,
            pdf_path=args.pdf_path,
            force_ocr=args.force_ocr,
            roll_up_groups=args.roll_up_groups,
            llm_enhanced_hierarchy=args.llm_enhanced_hierarchy,
        )
        logger.info("Received result:")
        print(json.dumps(result, indent=2))

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Script failed: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}") 