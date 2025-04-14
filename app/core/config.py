import os
import logging
from pydantic_settings import BaseSettings # Import from pydantic_settings


logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""
    DOCLING_MODELS_PATH: str = "./docling_models" # Default path relative to service root

    class Config:
        env_file = '.env' # Optional: load from a .env file
        env_file_encoding = 'utf-8'

settings = Settings()

# Ensure the models path exists after loading settings
if not os.path.exists(settings.DOCLING_MODELS_PATH) or not os.path.isdir(settings.DOCLING_MODELS_PATH):
    # This might be a critical error preventing the service from starting correctly.
    logger.error(f"Docling models path '{settings.DOCLING_MODELS_PATH}' does not exist or is not a directory.")
    # Depending on deployment strategy, you might want to raise an exception here
    # raise FileNotFoundError(f"Docling models path '{settings.DOCLING_MODELS_PATH}' not found.")
else:
    logger.info(f"Using Docling models path: {settings.DOCLING_MODELS_PATH}") 