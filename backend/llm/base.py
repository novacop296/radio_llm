"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.8 — LLM Provider Base Interface & Configuration

Module: base.py
Purpose:
- Abstract LLMProvider interface for pluggable language model backends.
- Configuration dataclass managing provider, model, temperature, and timeouts.
- LLMResponse container preserving metadata and raw outputs.

DISCLAIMER:
This module is a research prototype. Generated text and structured findings are
intermediate representations, NOT clinical diagnoses or verified medical facts.
"""

import abc
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class LLMConfig:
    """Configuration container for LLM providers."""
    provider: str = "mock"
    model: str = "mock-radiology-llm"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_seconds: int = 30
    retry_count: int = 3
    api_key: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load LLM configuration safely from environment variables without exposing secrets."""
        provider = os.getenv("LLM_PROVIDER", "mock").lower().strip()
        model = os.getenv("LLM_MODEL", "gpt-4o-mini" if provider == "openai" else "mock-radiology-llm")
        
        try:
            temp = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        except ValueError:
            temp = 0.0

        try:
            max_tok = int(os.getenv("LLM_MAX_TOKENS", "1024"))
        except ValueError:
            max_tok = 1024

        try:
            timeout = int(os.getenv("LLM_TIMEOUT", "30"))
        except ValueError:
            timeout = 30

        try:
            retries = int(os.getenv("LLM_RETRY_COUNT", "3"))
        except ValueError:
            retries = 3

        # API key retrieval without hardcoding
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

        return cls(
            provider=provider,
            model=model,
            temperature=temp,
            max_tokens=max_tok,
            timeout_seconds=timeout,
            retry_count=retries,
            api_key=api_key
        )


@dataclass
class LLMResponse:
    """Container for LLM output content and metadata."""
    content: str
    provider: str
    model: str
    raw_response: Optional[Any] = None
    execution_time_seconds: float = 0.0
    is_mock: bool = False
    tokens_used: Optional[int] = None


class LLMProvider(abc.ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text response from the language model backend.

        Args:
            prompt: User prompt containing sanitized structured evidence.
            system_prompt: System prompt defining research role and constraints.
            json_mode: Whether to enforce structured JSON output.
            **kwargs: Additional provider-specific parameters.

        Returns:
            LLMResponse object.
        """
        pass
