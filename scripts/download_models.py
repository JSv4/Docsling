import logging
import os
from pathlib import Path
import argparse # Use argparse for flexibility

# Import necessary libraries
try:
    import easyocr
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    # Import torch to check for CUDA/MPS availability if desired
    # import torch
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please ensure 'docling-core' and 'easyocr' are installed.")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ModelDownloader")


def download_docling_models(artifacts_path: str, force_download: bool = False) -> None:
    """
    Downloads the Docling layout models and specified EasyOCR models
    to the specified artifacts path.

    Args:
        artifacts_path (str): The directory where the models will be saved.
        force_download (bool): Whether to force download even if files exist.
    """
    target_path = Path(artifacts_path)
    target_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensuring Docling models are downloaded to '{target_path}'...")

    try:
        # Explicitly prefetch and download the Docling layout models
        StandardPdfPipeline.download_models_hf(
            local_dir=str(target_path),
            force_download=force_download # Use force_download flag
        )
        logger.info(f"Docling LAYOUT models download process complete for '{target_path}'.")
    except Exception as e:
        logger.error(f"Error downloading Docling layout models: {e}", exc_info=True)
        # Decide if you want to raise the error or continue to EasyOCR
        # raise e # Uncomment to make the build fail on Docling download error

    logger.info("Ensuring EasyOCR models are downloaded...")
    # Specify the languages you need. Add more as required.
    # Common: ['en']
    # Example from user: ['ch_tra', 'en']
    required_languages = ['en'] # MODIFY THIS LIST AS NEEDED

    # Check device availability (optional, for info)
    # gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
    # logger.info(f"GPU available for EasyOCR: {gpu_available}")

    try:
        # This initializes the Reader and triggers download for the specified languages
        # if they are not found in the model_storage_directory.
        # It will place models inside the artifacts_path.
        reader = easyocr.Reader(
            lang_list=required_languages,
            model_storage_directory=str(target_path),
            download_enabled=True, # Explicitly enable download
            # Set gpu=False if you know you won't have GPU during build/runtime,
            # or let it auto-detect. Build environments usually don't have GPUs.
            gpu=False
        )
        logger.info(f"EasyOCR models for languages {required_languages} download process complete for '{target_path}'.")
        # Clean up the reader object if necessary (might release memory)
        del reader
    except Exception as e:
        logger.error(f"Error downloading EasyOCR models: {e}", exc_info=True)
        # Decide if you want to raise the error
        # raise e # Uncomment to make the build fail on EasyOCR download error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Docling and EasyOCR models.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Directory to save the models.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if models seem to exist.",
    )

    args = parser.parse_args()

    # Use the provided path argument
    artifacts_path = args.path
    force = args.force

    logger.info(f"Starting model download process to path: {artifacts_path} (Force={force})")
    download_docling_models(artifacts_path, force_download=force)
    logger.info("Model download script finished.") 