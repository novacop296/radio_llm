"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.8 — Mock LLM Provider for Testing & Offline Execution

Module: mock_provider.py
Purpose:
- Deterministic MockLLMProvider simulating LLM report generation without requiring external API keys.
- Accurately generates structured finding statements and impression summaries adhering to evidence statuses.
- Used for comprehensive regression testing and local execution.
"""

import json
import re
import time
from typing import Dict, List, Optional, Any
from .base import LLMProvider, LLMConfig, LLMResponse


class MockLLMProvider(LLMProvider):
    """
    Deterministic Mock LLM Provider simulating structured report generation.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config or LLMConfig(provider="mock", model="mock-radiology-llm"))

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Parse the evidence from the user prompt and generate a compliant structured JSON response.
        """
        start_time = time.time()

        # Extract evidence dictionary from prompt text
        study_id = "Unknown"
        image_id = "Unknown"
        view = "Frontal"
        evidence_items = []

        try:
            # Look for JSON payload in prompt
            json_match = re.search(r"\{[\s\S]*\}", prompt)
            if json_match:
                payload = json.loads(json_match.group(0))
                study_id = payload.get("study_id", "Unknown")
                image_id = payload.get("image_id", "Unknown")
                view = payload.get("view", "Frontal")
                evidence_items = payload.get("evidence", [])
        except Exception:
            pass

        findings_output = []
        impression_items = []

        supported_findings = []
        possible_findings = []
        uncertain_findings = []
        absent_findings = []

        for item in evidence_items:
            name = item.get("finding", "Unknown")
            status = item.get("status", "uncertain").lower()
            location = item.get("location", "unspecified")
            severity = item.get("severity", "unspecified")

            # Format location string
            loc_str = f" in the {location.replace('_', ' ')}" if location and location != "unspecified" else ""
            # Format severity string
            sev_str = f"{severity} " if severity and severity != "unspecified" else ""

            if status == "supported":
                statement = f"{sev_str.capitalize()}{name} is present{loc_str}."
                supported_findings.append(f"{name}{loc_str}")
            elif status == "possible":
                statement = f"Possible {name.lower()}{loc_str} cannot be excluded; correlation is advised."
                possible_findings.append(f"Possible {name.lower()}{loc_str}")
            elif status == "absent":
                statement = f"No radiographic evidence of {name.lower()}."
                absent_findings.append(name)
            else:  # uncertain
                statement = f"Evaluation for {name.lower()} is indeterminate based on available visual features."
                uncertain_findings.append(name)

            findings_output.append({
                "finding": name,
                "statement": statement,
                "status": status,
                "location": location,
                "severity": severity
            })

        # Synthesize concise IMPRESSION section
        if supported_findings:
            for idx, sf in enumerate(supported_findings, start=1):
                impression_items.append(f"{idx}. {sf}.")
            if possible_findings:
                impression_items.append(f"{len(supported_findings) + 1}. {', '.join(possible_findings)}; clinical correlation advised.")
        elif possible_findings:
            impression_items.append(f"1. {', '.join(possible_findings)}; recommend clinical correlation.")
            if absent_findings:
                impression_items.append("2. No acute pneumothorax or pleural effusion.")
        elif absent_findings:
            impression_items.append("1. No acute cardiopulmonary abnormality.")
        else:
            impression_items.append("1. Unremarkable examination with indeterminate features.")

        report_dict = {
            "study_id": study_id,
            "image_id": image_id,
            "view": view,
            "findings": findings_output,
            "impression": impression_items,
            "metadata": {
                "provider": self.config.provider,
                "model": self.config.model,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        }

        content_str = json.dumps(report_dict, indent=2)
        exec_time = time.time() - start_time

        return LLMResponse(
            content=content_str,
            provider=self.config.provider,
            model=self.config.model,
            raw_response=report_dict,
            execution_time_seconds=round(exec_time, 4),
            is_mock=True,
            tokens_used=len(content_str.split())
        )
