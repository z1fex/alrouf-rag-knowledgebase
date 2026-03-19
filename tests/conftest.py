"""Shared test fixtures for the Alrouf RAG test suite."""

import os
import sys

import pytest

# Ensure mock mode is active for all tests
os.environ["USE_MOCK_LLM"] = "true"

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def documents_dir():
    """Path to the sample documents directory."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "documents")


@pytest.fixture(scope="session")
def mocks_dir():
    """Path to the mocks directory."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "mocks")
