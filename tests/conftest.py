# conftest.py
import os
import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_result = load_dotenv(override=True)

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment with a temporary directory for SQLite storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the directory with proper permissions
        storage_dir = Path(temp_dir) / "crewai_test_storage"
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate that the directory was created successfully
        if not storage_dir.exists() or not storage_dir.is_dir():
            raise RuntimeError(f"Failed to create test storage directory: {storage_dir}")
        
        # Verify directory permissions
        try:
            # Try to create a test file to verify write permissions
            test_file = storage_dir / ".permissions_test"
            test_file.touch()
            test_file.unlink()
        except (OSError, IOError) as e:
            raise RuntimeError(f"Test storage directory {storage_dir} is not writable: {e}")
        
        # Set environment variable to point to the test storage directory
        os.environ["CREWAI_STORAGE_DIR"] = str(storage_dir)
        
        yield
        
        # Cleanup is handled automatically when tempfile context exits
