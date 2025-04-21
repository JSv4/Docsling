import beam_client
import base64
import os
import argparse
import logging
import json
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def submit_pdf_to_beam(
    app_id: str,
    api_key: str,
    pdf_path: str,
    force_ocr: bool = False,
    roll_up_groups: bool = False,
    llm_enhanced_hierarchy: bool = False,
) -> Dict[str, Any]:
    """
    Submits a PDF file to a specified Beam endpoint for parsing.

    Args:
        app_id: The Beam Application ID.
        api_key: The Beam API Key.
        pdf_path: The local path to the PDF file to submit.
        force_ocr: Optional flag to force OCR processing.
        roll_up_groups: Optional flag to roll up groups.
        llm_enhanced_hierarchy: Optional flag to use LLM enhancement.

    Returns:
        A dictionary containing the response from the Beam endpoint.
    """
    logger.info(f"Preparing to submit PDF: {pdf_path}")

    # --- 1. Validate PDF Path ---
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found at path: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if not pdf_path.lower().endswith(".pdf"):
        logger.error(f"File is not a PDF: {pdf_path}")
        raise ValueError("File must be a PDF.")

    # --- 2. Read and Encode PDF ---
    try:
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
        logger.info("PDF read and base64 encoded successfully.")
    except Exception as e:
        logger.error(f"Failed to read or encode PDF: {e}")
        raise

    # --- 3. Construct Payload ---
    filename = os.path.basename(pdf_path)
    payload = {
        "pdf_base64": pdf_base64,
        "filename": filename,
        "force_ocr": force_ocr,
        "roll_up_groups": roll_up_groups,
        "llm_enhanced_hierarchy": llm_enhanced_hierarchy,
    }
    logger.info(f"Payload constructed for filename: {filename}")
    logger.debug(f"Payload options - force_ocr: {force_ocr}, roll_up_groups: {roll_up_groups}, llm_enhanced_hierarchy: {llm_enhanced_hierarchy}")


    # --- 4. Initialize Beam Client ---
    try:
        client = beam_client.BeamClient(app_id=app_id, api_key=api_key)
        logger.info(f"Beam client initialized for App ID: {app_id}")
    except Exception as e:
        logger.error(f"Failed to initialize Beam client: {e}")
        raise

    # --- 5. Trigger Beam Run ---
    logger.info("Triggering Beam run...")
    try:
        # Use client.trigger to start the run asynchronously
        triggered_run = client.trigger(payload=payload)
        run_id = triggered_run.run_id
        logger.info(f"Beam run triggered successfully. Run ID: {run_id}")

        # Wait for the run to complete and get the result
        # Note: For long-running tasks, you might implement polling or webhooks
        # instead of waiting synchronously. client.get() waits by default.
        logger.info("Waiting for Beam run to complete...")
        run_result = client.get(run_id=run_id) # Waits for completion

        if run_result and run_result.outputs:
             logger.info("Beam run completed successfully.")
             # Assuming the endpoint returns a dictionary directly
             return run_result.outputs
        elif run_result and run_result.status == beam_client.RunStatus.FAILED:
             logger.error(f"Beam run failed. Status: {run_result.status}")
             # You might want to inspect run_result.logs or other fields
             return {"error": f"Beam run failed with status: {run_result.status}", "run_id": run_id}
        else:
             logger.warning(f"Beam run finished with unexpected status or no output. Status: {run_result.status if run_result else 'N/A'}")
             return {"error": "Beam run finished with unexpected status or no output.", "run_id": run_id}

    except Exception as e:
        logger.error(f"An error occurred during Beam interaction: {e}")
        # Attempt to include run_id if available
        run_id_str = f" (Run ID: {run_id})" if 'run_id' in locals() else ""
        raise RuntimeError(f"Beam interaction failed{run_id_str}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit a PDF to a Beam endpoint for parsing.")

    # Arguments for Beam credentials (prefer environment variables)
    parser.add_argument(
        "--app-id",
        type=str,
        default=os.environ.get("BEAM_APP_ID"),
        help="Beam Application ID (can also be set via BEAM_APP_ID env var)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("BEAM_API_KEY"),
        help="Beam API Key (can also be set via BEAM_API_KEY env var)",
    )

    # Argument for the PDF file path
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the local PDF file to submit.",
    )

    # Optional arguments matching the endpoint's boolean flags
    parser.add_argument(
        "--force-ocr",
        action="store_true", # Makes it a flag, default is False
        help="Force OCR processing even if text layer exists.",
    )
    parser.add_argument(
        "--roll-up-groups",
        action="store_true",
        help="Enable roll-up groups feature.",
    )
    parser.add_argument(
        "--llm-enhanced-hierarchy",
        action="store_true",
        help="Enable LLM enhanced hierarchy feature.",
    )

    args = parser.parse_args()

    # Validate credentials are provided
    if not args.app_id:
        parser.error("Beam App ID is required. Set --app-id or BEAM_APP_ID environment variable.")
    if not args.api_key:
        parser.error("Beam API Key is required. Set --api-key or BEAM_API_KEY environment variable.")

    try:
        result = submit_pdf_to_beam(
            app_id=args.app_id,
            api_key=args.api_key,
            pdf_path=args.pdf_path,
            force_ocr=args.force_ocr,
            roll_up_groups=args.roll_up_groups,
            llm_enhanced_hierarchy=args.llm_enhanced_hierarchy,
        )
        logger.info("Received result from Beam:")
        # Pretty print the JSON result
        print(json.dumps(result, indent=2))

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Script failed: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}") 