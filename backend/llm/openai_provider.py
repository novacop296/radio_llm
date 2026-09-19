"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.8 — OpenAI LLM Provider Implementation

Module: openai_provider.py
Purpose:
- Integrates OpenAI-compatible chat completion endpoints.
- Implements deterministic temperature=0, response_format={"type": "json_object"}, timeout, and exponential retry.
- Uses standard library urllib.request to avoid mandatory heavy dependencies while supporting optional openai SDK.
"""

import json
import time
import urllib.request
import urllib.error
from typing import Dict, Optional, Any
from .base import LLMProvider, LLMConfig, LLMResponse


class OpenAIProvider(LLMProvider):
    """
    OpenAI-compatible LLM Provider.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config or LLMConfig.from_env())
        self.api_url = self.config.extra_params.get("api_url", "https://api.openai.com/v1/chat/completions")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Execute API request with timeout and exponential backoff.
        """
        api_key = self.config.api_key
        if not api_key:
            raise ValueError("[ERROR] OpenAI API key is missing. Set LLM_API_KEY or OPENAI_API_KEY environment variable.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.api_url, data=data_bytes, headers=headers, method="POST")

        last_error = None
        start_time = time.time()

        for attempt in range(1, self.config.retry_count + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                    res_body = response.read().decode("utf-8")
                    res_json = json.loads(res_body)

                    choice = res_json.get("choices", [{}])[0]
                    message_content = choice.get("message", {}).get("content", "")
                    tokens_used = res_json.get("usage", {}).get("total_tokens")

                    exec_time = time.time() - start_time
                    return LLMResponse(
                        content=message_content,
                        provider="openai",
                        model=self.config.model,
                        raw_response=res_json,
                        execution_time_seconds=round(exec_time, 4),
                        is_mock=False,
                        tokens_used=tokens_used
                    )
            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8") if e.fp else ""
                last_error = f"HTTPError {e.code}: {e.reason} - {error_body}"
                if e.code in [429, 500, 502, 503, 504]:
                    # Transient error, apply exponential backoff
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"OpenAI API error ({e.code}): {last_error}")
            except urllib.error.URLError as e:
                last_error = f"URLError: {e.reason}"
                time.sleep(2 ** attempt)
            except Exception as e:
                last_error = str(e)
                time.sleep(2 ** attempt)

        raise RuntimeError(f"OpenAI API request failed after {self.config.retry_count} attempts: {last_error}")
