"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.8 — LLM Provider Factory

Module: factory.py
Purpose:
- Instantiates the configured LLMProvider based on LLMConfig or environment variables.
"""

import os
from typing import Optional
from .base import LLMProvider, LLMConfig
from .mock_provider import MockLLMProvider
from .openai_provider import OpenAIProvider


def create_llm_provider(config: Optional[LLMConfig] = None) -> LLMProvider:
    """
    Factory function creating an LLMProvider instance.
    If no config is provided, configuration is loaded from environment variables.
    """
    cfg = config or LLMConfig.from_env()
    provider_name = cfg.provider.lower().strip()

    if provider_name in ["mock", "dummy", "test"]:
        return MockLLMProvider(cfg)
    elif provider_name in ["openai", "gpt"]:
        return OpenAIProvider(cfg)
    else:
        raise ValueError(f"Unsupported LLM provider: '{provider_name}'. Supported: ['mock', 'openai']")
