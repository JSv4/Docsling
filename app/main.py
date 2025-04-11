from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

app = FastAPI(
    title="Simple Microservice",
    description="A basic FastAPI microservice example.",
    version="0.1.0",
)

class HealthCheck(BaseModel):
    """Model for health check response."""
    status: str = "OK"

@app.get("/", tags=["General"])
async def read_root() -> Dict[str, str]:
    """
    Root endpoint providing a welcome message.

    Returns:
        A dictionary containing a welcome message.
    """
    return {"message": "Welcome to the Simple Microservice!"}

@app.get("/health", tags=["General"], response_model=HealthCheck)
async def health_check() -> HealthCheck:
    """
    Health check endpoint to verify service status.

    Returns:
        A HealthCheck object indicating the service status.
    """
    return HealthCheck(status="OK")

# You can add more endpoints here as your microservice grows 