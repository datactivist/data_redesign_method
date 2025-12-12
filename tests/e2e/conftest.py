"""
Pytest configuration and fixtures for E2E tests.
"""

import pytest
import sys
from pathlib import Path

# Add package to Python path
PACKAGE_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

# Test data paths
TEST_DATA_DIR = PACKAGE_ROOT / "test_data"
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


@pytest.fixture(scope="session")
def package_root():
    """Return the package root directory."""
    return PACKAGE_ROOT


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the test data directory."""
    return TEST_DATA_DIR


@pytest.fixture(scope="session")
def artifacts_dir():
    """Return the artifacts output directory."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


@pytest.fixture
def output_dir(artifacts_dir, request):
    """Create a unique output directory for each test."""
    test_name = request.node.name
    output = artifacts_dir / test_name
    output.mkdir(parents=True, exist_ok=True)
    return output
