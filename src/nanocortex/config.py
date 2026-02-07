"""System configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class LLMProviderConfig(BaseModel):
    api_key: str = ""
    model: str = ""
    base_url: str = ""
    role: str = ""  # "orchestrator" | "auditor" | "ingestion"


class Settings(BaseModel):
    # LLM Providers
    orchestrator: LLMProviderConfig = Field(default_factory=lambda: LLMProviderConfig(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        base_url="https://api.openai.com/v1",
        role="orchestrator",
    ))
    auditor: LLMProviderConfig = Field(default_factory=lambda: LLMProviderConfig(
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        base_url="https://api.anthropic.com",
        role="auditor",
    ))
    ingestion_helper: LLMProviderConfig = Field(default_factory=lambda: LLMProviderConfig(
        api_key=os.getenv("KIMIK_API_KEY", ""),
        model=os.getenv("KIMIK_MODEL", "kimik-2.5"),
        base_url=os.getenv("KIMIK_BASE_URL", ""),
        role="ingestion",
    ))

    # System paths
    audit_dir: Path = Path(os.getenv("AUDIT_DIR", "./data/audit"))
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))

    # Feature flags
    enable_human_in_loop: bool = os.getenv("ENABLE_HUMAN_IN_LOOP", "true").lower() == "true"
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


def get_settings() -> Settings:
    return Settings()
