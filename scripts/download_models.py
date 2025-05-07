import logging
import os
import argparse
from pathlib import Path
import sys # Import sys for exit
from typing import Optional

try:
    import easyocr
    # Import the new recommended download utility
    from docling.utils.model_downloader import download_models as download_docling_core_models
    from huggingface_hub import HfFolder
except ImportError as e:
    print(f"Error importing necessary libraries: {e}")
    print("Please ensure docling-core, easyocr, and huggingface-hub are installed.")
    sys.exit(1) # Use sys.exit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def download_all_models(
    artifacts_path: str,
    force_download: bool = False,
    hf_token: Optional[str] = None # Add hf_token parameter
) -> None:
    """
    Downloads the core Docling models and specified EasyOCR models
    to the specified artifacts path using the latest methods.

    Args:
        artifacts_path (str): The base directory where models will be saved.
                              Core models go here, EasyOCR models go into a subdirectory.
        force_download (bool): Whether to force download core Docling models.
        hf_token (Optional[str]): Hugging Face API token for authentication.
    """
    # Base artifacts path
    base_target_path = Path(artifacts_path)
    base_target_path.mkdir(parents=True, exist_ok=True)
    resolved_base_path = str(base_target_path.resolve())
    logger.info(f"Ensured base models directory exists: '{resolved_base_path}'")

    # --- Handle Hugging Face Token ---
    token_to_use = hf_token # Use token from arg if provided
    if token_to_use:
        logger.info("Using Hugging Face token provided via command-line argument.")
        # Set the environment variable for huggingface_hub library to pick up
        os.environ['HUGGING_FACE_HUB_TOKEN'] = token_to_use
    else:
        # Check if already logged in via CLI or env var is set elsewhere
        token_to_use = HfFolder.get_token() # Check stored token
        if token_to_use:
             logger.info("Using Hugging Face token found via huggingface-cli login or HUGGING_FACE_HUB_TOKEN env var.")
        else:
             logger.warning("No Hugging Face token provided or found. Download might fail if core models require authentication.")
             logger.warning("Download might fail if models require authentication.")
             # No need to explicitly set env var if token is None

    # --- Download Core Docling Models ---
    logger.info("Attempting to download core Docling models (layout, tableformer, etc.)...")
    try:
        # Set DOCLING_ARTIFACTS_PATH environment variable BEFORE calling download
        logger.info(f"Setting DOCLING_ARTIFACTS_PATH environment variable to: {resolved_base_path}")
        os.environ['DOCLING_ARTIFACTS_PATH'] = resolved_base_path

        # Call the download function WITHOUT local_dir, it uses the env var
        # It will use the HUGGING_FACE_HUB_TOKEN env var if we set it above
        download_docling_core_models(output_dir=base_target_path, force=force_download)

        logger.info(f"Core Docling models download process completed for '{resolved_base_path}'.")
        # Optional: Verify a key file exists within the resolved_path
        # Example: layout_model_path = target_path / "layout" / "model.safetensors" # Adjust if needed
        # if layout_model_path.exists(): logger.info("Verified layout model component.")

    except ImportError:
         logger.error("Failed to import docling.utils.model_downloader. Is docling-core installed and up-to-date?")
    except TypeError as te:
         logger.error(f"TypeError calling download_docling_core_models: {te}. Maybe the 'force' argument name changed?")
    except Exception as e:
        # Check specifically for authentication errors if possible
        if "401 Client Error" in str(e) or "Unauthorized" in str(e):
             logger.error(f"Hugging Face authentication failed: {e}")
             logger.error("Ensure the provided token is valid and has read access, or log in using 'huggingface-cli login'.")
        else:
             logger.error(f"An error occurred during core Docling model download: {e}", exc_info=True)
    finally:
        # Clean up environment variables set by this script
        if 'DOCLING_ARTIFACTS_PATH' in os.environ:
            del os.environ['DOCLING_ARTIFACTS_PATH']
        if hf_token and 'HUGGING_FACE_HUB_TOKEN' in os.environ: # Only delete if we set it
             del os.environ['HUGGING_FACE_HUB_TOKEN']

    # --- Download EasyOCR Models ---
    logger.info("Attempting to download EasyOCR models...")
    try:
        easyocr_subdir = base_target_path / "EasyOcr"
        easyocr_subdir.mkdir(parents=True, exist_ok=True)
        resolved_easyocr_path = str(easyocr_subdir.resolve())
        logger.info(f"EasyOCR models will be downloaded to: '{resolved_easyocr_path}'")

        # We need codes that trigger the download of the required model files
        # 'en' covers English, and 'fr' (or 'es', 'de', etc.) should cover the latin_g2 model.
        required_easyocr_langs = ['en', 'fr', 'la']
        logger.info(f"Downloading EasyOCR models for languages: {required_easyocr_langs}")

        reader = easyocr.Reader(
            required_easyocr_langs, # Use the updated list
            model_storage_directory=resolved_easyocr_path,
            user_network_directory=resolved_easyocr_path,
            verbose=True,
            download_enabled=True
        )
        logger.info(f"EasyOCR models download process completed for '{resolved_easyocr_path}'.")
        del reader
    except ValueError as ve:
        logger.error(f"ValueError during EasyOCR initialization: {ve}. Check if language codes are supported.")
    except Exception as e:
        logger.error(f"An error occurred during EasyOCR model download: {e}", exc_info=True)

    logger.info("Model download process finished.")


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Download Docling and EasyOCR models.")
    parser.add_argument(
        "--path",
        type=str,
        default="./docling_models",
        help="Base directory to save models (defaults to ./docling_models). EasyOCR models go into an 'EasyOcr' subdirectory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download for core Docling models (EasyOCR downloads if missing).",
    )
    # Add the new token argument
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None, # Default to None if not provided
        help="Hugging Face API token for downloading private/gated models.",
    )

    args = parser.parse_args()

    artifacts_directory = args.path
    force_flag = args.force
    token_flag = args.hf_token # Get the token from args

    print(f"--- Starting Docling Model Download Script ---")
    resolved_artifacts_directory = str(Path(artifacts_directory).resolve())
    print(f"Target directory (Base): {resolved_artifacts_directory}")
    print(f"Target directory (EasyOCR): {Path(resolved_artifacts_directory) / 'EasyOcr'}") # Show the specific EasyOCR path
    print(f"Force download (Core Docling): {force_flag}")
    if token_flag:
        print(f"Hugging Face Token: Provided via argument (value hidden)")
    else:
        print(f"Hugging Face Token: Not provided via argument (will check login/env var)")

    # Call the main function, passing the token
    download_all_models(
        artifacts_directory,
        force_download=force_flag,
        hf_token=token_flag # Pass the token here
    )

    print(f"--- Finished Docling Model Download Script ---") 