"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.8 — Report Generator Coordinator

Module: report_generator.py
Purpose:
- Coordinates the complete LLM report generation pipeline:
  1. Ingests sanitized LLM Input Package from Evidence Layer.
  2. Builds deterministic system and user prompts with safety constraints.
  3. Invokes configured LLMProvider (Mock or Real).
  4. Parses structured JSON output.
  5. Validates report safety, schema compliance, and evidence traceability via validate_report.py.
  6. Returns validated report dictionary with execution metadata.

DISCLAIMER:
This module is a research prototype. Generated report statements are pattern representations,
NOT clinical diagnoses or verified medical facts.
"""

import json
import re
from typing import Dict, Any, Optional, Tuple
from llm.base import LLMProvider, LLMConfig, LLMResponse
from llm.factory import create_llm_provider
from llm.prompts import build_system_prompt, build_user_prompt
from validate_report import validate_generated_report


class RadiologyReportGenerator:
    """
    Coordinator class for structured radiology report generation from Evidence Layer packages.
    """

    def __init__(self, provider: Optional[LLMProvider] = None, config: Optional[LLMConfig] = None):
        self.provider = provider or create_llm_provider(config)

    def generate_report(
        self,
        llm_input_package: Dict[str, Any],
        enforce_validation: bool = True
    ) -> Tuple[Dict[str, Any], LLMResponse, Dict[str, Any]]:
        """
        Generate a validated, structured radiology report.

        Args:
            llm_input_package: Sanitized dictionary from EvidenceLayer.build_llm_input().
            enforce_validation: Whether to raise ValueError on safety validation failure.

        Returns:
            Tuple of:
            - validated_report: Dict adhering to docs/report_schema.json
            - raw_response: LLMResponse object
            - validation_report: Dict containing 'is_valid', 'issues'
        """
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(llm_input_package)

        # Call LLM Provider
        response = self.provider.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True
        )

        # Parse JSON from response content
        raw_text = response.content.strip()
        try:
            # Handle potential markdown code blocks like ```json ... ```
            clean_text = raw_text
            if "```" in clean_text:
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
                if match:
                    clean_text = match.group(1)

            parsed_report = json.loads(clean_text)
        except json.JSONDecodeError as e:
            err_msg = f"Failed to parse LLM JSON response: {e}\nRaw output: {raw_text}"
            if enforce_validation:
                raise ValueError(err_msg)
            return {"error": "JSONDecodeError", "raw_output": raw_text}, response, {"is_valid": False, "issues": [err_msg]}

        # Validate Report
        is_valid, issues = validate_generated_report(parsed_report, llm_input_package)
        validation_report = {
            "is_valid": is_valid,
            "issues": issues
        }

        if not is_valid and enforce_validation:
            raise ValueError(f"Report validation failed with {len(issues)} safety violations:\n" + "\n".join(f"- {iss}" for iss in issues))

        # Attach metadata if not present
        if "metadata" not in parsed_report or not parsed_report["metadata"]:
            parsed_report["metadata"] = {
                "provider": response.provider,
                "model": response.model,
                "execution_time_seconds": response.execution_time_seconds,
                "is_mock": response.is_mock
            }

        return parsed_report, response, validation_report
