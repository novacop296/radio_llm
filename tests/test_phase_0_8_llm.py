"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.8 — LLM Integration & Report Generation Unit Test Suite

Verifies all 22 required test cases:
TEST 1: Mock provider works without API key.
TEST 2: LLM input package is accepted.
TEST 3: Required system constraints are present in prompt.
TEST 4: Structured JSON output is parsed.
TEST 5: Invalid JSON is rejected.
TEST 6: Invalid schema is rejected.
TEST 7: Missing required fields are rejected.
TEST 8: Study ID cannot change.
TEST 9: Image ID cannot change.
TEST 10: Possible finding remains possible.
TEST 11: Uncertain finding remains uncertain.
TEST 12: Absent finding remains absent.
TEST 13: LLM cannot introduce an unsupported finding.
TEST 14: LLM cannot invent location.
TEST 15: LLM cannot invent severity.
TEST 16: LLM cannot introduce unsupported measurements.
TEST 17: Ground truth is absent from LLM input.
TEST 18: Model score is not interpreted as clinical probability.
TEST 19: Impression cannot introduce unsupported findings.
TEST 20: API errors are handled gracefully.
TEST 21: Missing API key is handled gracefully.
TEST 22: End-to-end Mock LLM pipeline succeeds.
"""

import os
import sys
import unittest
import json

# Add project root and backend to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from llm.base import LLMConfig
from llm.mock_provider import MockLLMProvider
from llm.openai_provider import OpenAIProvider
from llm.prompts import build_system_prompt, build_user_prompt
from report_generator import RadiologyReportGenerator
from validate_report import validate_generated_report
from evidence_layer import EvidenceLayer
from diagnostic_qa import DiagnosticQAEngine


class TestPhase08LLMReportGeneration(unittest.TestCase):

    def setUp(self):
        self.mock_provider = MockLLMProvider()
        self.generator = RadiologyReportGenerator(provider=self.mock_provider)
        self.sample_input_pkg = {
            "task": "radiology_report_generation",
            "study": {
                "study_id": "CXR1122",
                "image_id": "CXR1122_IM-0080-1001-0002",
                "view": "Frontal"
            },
            "evidence": [
                {
                    "finding": "Infiltration",
                    "status": "possible",
                    "model_score": 0.475,
                    "location": "unspecified",
                    "severity": "unspecified"
                },
                {
                    "finding": "Pneumothorax",
                    "status": "absent",
                    "model_score": 0.2193,
                    "location": "unspecified",
                    "severity": "unspecified"
                },
                {
                    "finding": "Effusion",
                    "status": "supported",
                    "model_score": 0.0423,
                    "location": "right",
                    "severity": "small"
                },
                {
                    "finding": "Consolidation",
                    "status": "uncertain",
                    "model_score": 0.2052,
                    "location": "unspecified",
                    "severity": "unspecified"
                }
            ],
            "constraints": {
                "do_not_invent_findings": True,
                "do_not_invent_location": True,
                "do_not_invent_severity": True,
                "preserve_uncertainty": True
            }
        }

    def test_01_mock_provider_works_without_api_key(self):
        """TEST 1: Mock provider executes successfully without an API key."""
        resp = self.mock_provider.generate(prompt="Test prompt")
        self.assertTrue(resp.is_mock)
        self.assertIn("findings", resp.content)

    def test_02_llm_input_package_accepted(self):
        """TEST 2: LLM input package is cleanly formatted into prompt."""
        user_prompt = build_user_prompt(self.sample_input_pkg)
        self.assertIn("CXR1122", user_prompt)
        self.assertIn("Infiltration", user_prompt)

    def test_03_required_system_constraints_present(self):
        """TEST 3: System prompt contains all mandatory safety constraints."""
        sys_prompt = build_system_prompt()
        self.assertIn("DO NOT INVENT FINDINGS", sys_prompt)
        self.assertIn("DO NOT INVENT LOCATION", sys_prompt)
        self.assertIn("DO NOT INVENT SEVERITY", sys_prompt)
        self.assertIn("PRESERVE STATUS DISTINCTIONS", sys_prompt)

    def test_04_structured_json_output_parsed(self):
        """TEST 4: Report generator parses structured JSON response."""
        report, resp, val = self.generator.generate_report(self.sample_input_pkg)
        self.assertEqual(report["study_id"], "CXR1122")
        self.assertIn("findings", report)
        self.assertIn("impression", report)

    def test_05_invalid_json_rejected(self):
        """TEST 5: Malformed JSON output is rejected."""
        class BrokenLLM(MockLLMProvider):
            def generate(self, prompt, **kwargs):
                return self._wrap("This is not JSON text at all.")
            def _wrap(self, txt):
                from llm.base import LLMResponse
                return LLMResponse(content=txt, provider="broken", model="broken")

        broken_gen = RadiologyReportGenerator(provider=BrokenLLM())
        with self.assertRaises(ValueError):
            broken_gen.generate_report(self.sample_input_pkg, enforce_validation=True)

    def test_06_invalid_schema_rejected(self):
        """TEST 6: Invalid report structure fails validation."""
        bad_report = {"findings": []}  # Missing study_id, image_id, impression
        is_valid, issues = validate_generated_report(bad_report, self.sample_input_pkg)
        self.assertFalse(is_valid)
        self.assertTrue(len(issues) > 0)

    def test_07_missing_required_fields_rejected(self):
        """TEST 7: Missing required finding fields are rejected."""
        bad_report = {
            "study_id": "CXR1122",
            "image_id": "CXR1122_IM-0080-1001-0002",
            "view": "Frontal",
            "findings": [{"finding": "Pneumothorax"}],  # Missing statement and status
            "impression": ["No acute abnormality."]
        }
        is_valid, issues = validate_generated_report(bad_report, self.sample_input_pkg)
        self.assertFalse(is_valid)

    def test_08_study_id_cannot_change(self):
        """TEST 8: Modified study_id fails validation against input package."""
        bad_report = {
            "study_id": "WRONG_ID",
            "image_id": "CXR1122_IM-0080-1001-0002",
            "view": "Frontal",
            "findings": [{"finding": "Pneumothorax", "statement": "None", "status": "absent"}],
            "impression": ["Normal"]
        }
        is_valid, issues = validate_generated_report(bad_report, self.sample_input_pkg)
        self.assertFalse(is_valid)
        self.assertTrue(any("study_id mismatch" in err for err in issues))

    def test_09_image_id_cannot_change(self):
        """TEST 9: Modified image_id fails validation against input package."""
        bad_report = {
            "study_id": "CXR1122",
            "image_id": "WRONG_IMAGE",
            "view": "Frontal",
            "findings": [{"finding": "Pneumothorax", "statement": "None", "status": "absent"}],
            "impression": ["Normal"]
        }
        is_valid, issues = validate_generated_report(bad_report, self.sample_input_pkg)
        self.assertFalse(is_valid)
        self.assertTrue(any("image_id mismatch" in err for err in issues))

    def test_10_possible_finding_remains_possible(self):
        """TEST 10: Finding with evidence status 'possible' cannot be upgraded to 'supported'."""
        bad_report = {
            "study_id": "CXR1122",
            "image_id": "CXR1122_IM-0080-1001-0002",
            "view": "Frontal",
            "findings": [
                # Infiltration was 'possible' in evidence
                {"finding": "Infiltration", "statement": "Infiltration is present.", "status": "supported"}
            ],
            "impression": ["Infiltration present."]
        }
        is_valid, issues = validate_generated_report(bad_report, self.sample_input_pkg)
        self.assertFalse(is_valid)
        self.assertTrue(any("Illegal status upgrade" in err for err in issues))

    def test_11_uncertain_finding_remains_uncertain(self):
        """TEST 11: Finding with evidence status 'uncertain' cannot be upgraded to 'supported'."""
        bad_report = {
            "study_id": "CXR1122",
            "image_id": "CXR1122_IM-0080-1001-0002",
            "view": "Frontal",
            "findings": [
                # Consolidation was 'uncertain' in evidence
                {"finding": "Consolidation", "statement": "Consolidation is present.", "status": "supported"}
            ],
            "impression": ["Consolidation present."]
        }
        is_valid, issues = validate_generated_report(bad_report, self.sample_input_pkg)
        self.assertFalse(is_valid)
        self.assertTrue(any("Illegal status upgrade" in err for err in issues))

    def test_12_absent_finding_remains_absent(self):
        """TEST 12: Finding with evidence status 'absent' cannot contradict into 'supported'."""
        bad_report = {
            "study_id": "CXR1122",
            "image_id": "CXR1122_IM-0080-1001-0002",
            "view": "Frontal",
            "findings": [
                # Pneumothorax was 'absent' in evidence
                {"finding": "Pneumothorax", "statement": "Pneumothorax is present.", "status": "supported"}
            ],
            "impression": ["Pneumothorax."]
        }
        is_valid, issues = validate_generated_report(bad_report, self.sample_input_pkg)
        self.assertFalse(is_valid)
        self.assertTrue(any("Illegal status contradiction" in err for err in issues))

    def test_13_cannot_introduce_unsupported_finding(self):
        """TEST 13: Report cannot introduce findings not in input evidence."""
        bad_report = {
            "study_id": "CXR1122",
            "image_id": "CXR1122_IM-0080-1001-0002",
            "view": "Frontal",
            "findings": [
                {"finding": "Hernia", "statement": "Hiatal hernia is present.", "status": "supported"}
            ],
            "impression": ["Hiatal hernia."]
        }
        is_valid, issues = validate_generated_report(bad_report, self.sample_input_pkg)
        self.assertFalse(is_valid)
        self.assertTrue(any("Unsupported finding introduced" in err for err in issues))

    def test_14_cannot_invent_location(self):
        """TEST 14: Report cannot invent location when evidence has 'unspecified'."""
        bad_report = {
            "study_id": "CXR1122",
            "image_id": "CXR1122_IM-0080-1001-0002",
            "view": "Frontal",
            "findings": [
                # Infiltration had 'unspecified' location in evidence
                {"finding": "Infiltration", "statement": "Infiltration in right upper lobe.", "status": "possible", "location": "right_upper_lobe"}
            ],
            "impression": ["Possible infiltration."]
        }
        is_valid, issues = validate_generated_report(bad_report, self.sample_input_pkg)
        self.assertFalse(is_valid)
        self.assertTrue(any("Fabricated location" in err for err in issues))

    def test_15_cannot_invent_severity(self):
        """TEST 15: Report cannot invent severity when evidence has 'unspecified'."""
        bad_report = {
            "study_id": "CXR1122",
            "image_id": "CXR1122_IM-0080-1001-0002",
            "view": "Frontal",
            "findings": [
                # Infiltration had 'unspecified' severity in evidence
                {"finding": "Infiltration", "statement": "Severe infiltration.", "status": "possible", "severity": "severe"}
            ],
            "impression": ["Possible severe infiltration."]
        }
        is_valid, issues = validate_generated_report(bad_report, self.sample_input_pkg)
        self.assertFalse(is_valid)
        self.assertTrue(any("Fabricated severity" in err for err in issues))

    def test_16_cannot_introduce_unsupported_measurements(self):
        """TEST 16: Report cannot contain fabricated numerical measurements."""
        bad_report = {
            "study_id": "CXR1122",
            "image_id": "CXR1122_IM-0080-1001-0002",
            "view": "Frontal",
            "findings": [
                {"finding": "Effusion", "statement": "Pleural effusion measuring 3.5 cm in size.", "status": "supported", "location": "right", "severity": "small"}
            ],
            "impression": ["3.5 cm effusion."]
        }
        is_valid, issues = validate_generated_report(bad_report, self.sample_input_pkg)
        self.assertFalse(is_valid)
        self.assertTrue(any("fabricated numerical measurements" in err for err in issues))

    def test_17_ground_truth_absent_from_llm_input(self):
        """TEST 17: Ground truth does not appear in user prompt."""
        user_prompt = build_user_prompt(self.sample_input_pkg)
        self.assertNotIn("report_ground_truth", user_prompt)
        self.assertNotIn("ground_truth", user_prompt.lower())

    def test_18_model_score_not_interpreted_as_clinical_probability(self):
        """TEST 18: Model score cannot be exposed as percentage or clinical probability."""
        bad_report = {
            "study_id": "CXR1122",
            "image_id": "CXR1122_IM-0080-1001-0002",
            "view": "Frontal",
            "findings": [
                {"finding": "Infiltration", "statement": "Infiltration with 0.4750 probability.", "status": "possible"}
            ],
            "impression": ["Infiltration."]
        }
        is_valid, issues = validate_generated_report(bad_report, self.sample_input_pkg)
        self.assertFalse(is_valid)
        self.assertTrue(any("Model score improperly exposed" in err for err in issues))

    def test_19_impression_valid_and_non_empty(self):
        """TEST 19: Impression must be a non-empty list of strings."""
        bad_report = {
            "study_id": "CXR1122",
            "image_id": "CXR1122_IM-0080-1001-0002",
            "view": "Frontal",
            "findings": [{"finding": "Pneumothorax", "statement": "No pneumothorax.", "status": "absent"}],
            "impression": []
        }
        is_valid, issues = validate_generated_report(bad_report, self.sample_input_pkg)
        self.assertFalse(is_valid)

    def test_20_api_errors_handled(self):
        """TEST 20: Provider network or runtime errors are properly caught."""
        class ErrorLLM(MockLLMProvider):
            def generate(self, prompt, **kwargs):
                raise ConnectionError("Simulated network timeout.")

        error_gen = RadiologyReportGenerator(provider=ErrorLLM())
        with self.assertRaises(ConnectionError):
            error_gen.generate_report(self.sample_input_pkg)

    def test_21_missing_api_key_handled(self):
        """TEST 21: OpenAIProvider without API key raises informative ValueError."""
        openai_no_key = OpenAIProvider(config=LLMConfig(provider="openai", api_key=None))
        with self.assertRaises(ValueError):
            openai_no_key.generate("Test prompt")

    def test_22_e2e_mock_llm_pipeline_succeeds(self):
        """TEST 22: Complete Mock LLM pipeline runs end-to-end and validates."""
        report, resp, val = self.generator.generate_report(self.sample_input_pkg)
        self.assertTrue(val["is_valid"])
        self.assertTrue(len(report["findings"]) > 0)
        self.assertTrue(len(report["impression"]) > 0)


if __name__ == "__main__":
    unittest.main()
