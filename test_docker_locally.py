#!/usr/bin/env python3
"""
Builds the Docker image for the Docling Parser service, runs it locally,
and tests the /parse/ endpoint.
"""
import subprocess
import time
import requests
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# --- Configuration ---
IMAGE_NAME: str = "docling-parser-local-test"
CONTAINER_NAME: str = "docling-parser-test-container"
HOST_PORT: int = 8000 # Host port to map to container's port
CONTAINER_PORT: int = 8000 # Port the app runs on inside the container
PROJECT_ROOT: Path = Path(__file__).parent.resolve()

# Path to the sample PDF file, assuming this script is in the project root
# This matches the structure used in tests/fixtures.py
SAMPLE_PDF_PATH: Path = PROJECT_ROOT / "tests" / "fixtures" / "EtonPharmaceuticalsInc_20191114_10-Q_EX-10.1_11893941_EX-10.1_Development_Agreement_ZrZJLLv.pdf"

HEALTH_ENDPOINT_URL: str = f"http://localhost:{HOST_PORT}/health"
PARSE_ENDPOINT_URL: str = f"http://localhost:{HOST_PORT}/parse/"

# --- Helper Functions ---

def run_command(command: list[str], cwd: Optional[str | Path] = None) -> Tuple[bool, str, str]:
    """
    Runs a shell command and returns success status, stdout, and stderr.

    Args:
        command: The command and its arguments as a list of strings.
        cwd: The working directory for the command.

    Returns:
        A tuple (success, stdout, stderr).
    """
    print(f"\n▶️ Running command: {' '.join(command)}")
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False, # Don't raise exception for non-zero exit codes immediately
            cwd=cwd or PROJECT_ROOT
        )
        if process.returncode == 0:
            print(f"✅ Command successful: {' '.join(command)}")
            if process.stdout:
                print("Stdout:\n", process.stdout.strip())
            return True, process.stdout, process.stderr
        else:
            print(f"❌ Command failed with exit code {process.returncode}: {' '.join(command)}")
            if process.stdout:
                print("Stdout:\n", process.stdout.strip())
            if process.stderr:
                print("Stderr:\n", process.stderr.strip())
            return False, process.stdout, process.stderr
    except FileNotFoundError:
        print(f"❌ Command not found: {command[0]}. Is Docker installed and in PATH?")
        return False, "", f"Command not found: {command[0]}"
    except Exception as e:
        print(f"❌ An unexpected error occurred while running command: {' '.join(command)}")
        print(f"Error: {e}")
        return False, "", str(e)

def build_docker_image() -> bool:
    """Builds the Docker image from the Dockerfile."""
    print("\n--- Building Docker Image ---")
    command = [
        "docker", "build",
        "-t", IMAGE_NAME,
        "." # Docker context path
    ]
    success, _, _ = run_command(command)
    return success

def run_docker_container() -> bool:
    """Runs the Docker container in detached mode."""
    print("\n--- Running Docker Container ---")
    # Ensure no container with the same name is already running
    stop_and_remove_container(silent=True)

    command = [
        "docker", "run",
        "-d", # Detached mode
        "-p", f"{HOST_PORT}:{CONTAINER_PORT}",
        "--name", CONTAINER_NAME,
        IMAGE_NAME
    ]
    success, _, _ = run_command(command)
    if success:
        print(f"Container {CONTAINER_NAME} started. Waiting for app to be healthy...")
        return wait_for_health_check()
    return False

def wait_for_health_check(timeout_seconds: int = 60, interval_seconds: int = 5) -> bool:
    """
    Waits for the application's health check endpoint to return a successful response.

    Args:
        timeout_seconds: Maximum time to wait for the health check.
        interval_seconds: Time interval between health check attempts.

    Returns:
        True if the health check succeeds within the timeout, False otherwise.
    """
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            response = requests.get(HEALTH_ENDPOINT_URL, timeout=5)
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("status") == "ok":
                        print(f"✅ Health check passed: {data.get('message', 'Status OK')}")
                        return True
                    else:
                        print(f"🟡 Health check status not 'ok': {data}")
                except json.JSONDecodeError:
                    print(f"🟡 Health check returned 200 but response was not valid JSON: {response.text}")
            else:
                print(f"🟡 Health check attempt failed with status {response.status_code}. Retrying...")
        except requests.exceptions.ConnectionError:
            print("🟡 Health check: Connection refused. Service might still be starting. Retrying...")
        except requests.exceptions.RequestException as e:
            print(f"🟡 Health check attempt failed with error: {e}. Retrying...")

        time.sleep(interval_seconds)

    print(f"❌ Health check timed out after {timeout_seconds} seconds.")
    return False

def test_parse_endpoint() -> bool:
    """
    Tests the /parse/ endpoint of the running container.
    """
    print("\n--- Testing /parse/ Endpoint ---")
    if not SAMPLE_PDF_PATH.exists():
        print(f"❌ Test PDF file not found at: {SAMPLE_PDF_PATH}")
        return False

    try:
        with open(SAMPLE_PDF_PATH, "rb") as f:
            files = {"file": (SAMPLE_PDF_PATH.name, f, "application/pdf")}
            # These are form data fields, not JSON payload
            data = {
                "force_ocr": False,
                "roll_up_groups": True, # Example option
                "llm_enhanced_hierarchy": False # Example option
            }
            print(f"Sending POST request to {PARSE_ENDPOINT_URL} with {SAMPLE_PDF_PATH.name}...")
            print(f"Form data: {data}")

            # Add a reasonable timeout for parsing
            response = requests.post(PARSE_ENDPOINT_URL, files=files, data=data, timeout=180)

            print(f"Response Status Code: {response.status_code}")
            if response.status_code == 200:
                print("✅ /parse/ endpoint test successful!")
                try:
                    response_data = response.json()
                    print("Response JSON (first 500 chars):")
                    print(json.dumps(response_data, indent=2)[:500] + "...")
                    # Add more specific assertions here if needed, e.g.:
                    # assert "title" in response_data
                    # assert response_data["pageCount"] > 0
                    return True
                except json.JSONDecodeError:
                    print("❌ Response is not valid JSON.")
                    print("Response Text:")
                    print(response.text)
                    return False
            else:
                print("❌ /parse/ endpoint test failed.")
                print("Response Text:")
                print(response.text)
                return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Request to /parse/ endpoint failed: {e}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred during /parse/ test: {e}")
        return False

def stop_and_remove_container(silent: bool = False) -> None:
    """Stops and removes the Docker container."""
    if not silent:
        print("\n--- Stopping and Removing Docker Container ---")
    run_command(["docker", "stop", CONTAINER_NAME])
    run_command(["docker", "rm", CONTAINER_NAME])
    if not silent:
        print(f"Container {CONTAINER_NAME} stopped and removed.")

# --- Main Execution ---
def main():
    """Main script execution flow."""
    overall_success = False
    try:
        if not build_docker_image():
            print("\n❌ Docker image build failed. Exiting.")
            return

        if not run_docker_container():
            print("\n❌ Docker container failed to run or become healthy. Exiting.")
            return

        if test_parse_endpoint():
            print("\n✅ All tests passed successfully!")
            overall_success = True
        else:
            print("\n❌ Some tests failed.")

    except Exception as e:
        print(f"\n❌ An unexpected error occurred in main execution: {e}")
    finally:
        stop_and_remove_container()
        if overall_success:
            print("\n🎉 Script finished successfully.")
        else:
            print("\n💔 Script finished with errors.")

if __name__ == "__main__":
    if not SAMPLE_PDF_PATH.exists():
        print(f"CRITICAL ERROR: Sample PDF file not found at the expected location: {SAMPLE_PDF_PATH}")
        print("Please ensure the PDF file exists and the SAMPLE_PDF_PATH variable is correct.")
    else:
        main() 