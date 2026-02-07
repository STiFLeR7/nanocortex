"""Shared pytest fixtures for nanocortex tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from nanocortex.audit.logger import AuditLogger
from nanocortex.config import Settings


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def audit_logger(temp_dir: Path) -> AuditLogger:
    """Create an audit logger with a temp directory."""
    return AuditLogger(temp_dir / "audit")


@pytest.fixture
def settings(temp_dir: Path) -> Settings:
    """Create test settings with temp directories."""
    return Settings(
        audit_dir=temp_dir / "audit",
        data_dir=temp_dir / "data",
    )
