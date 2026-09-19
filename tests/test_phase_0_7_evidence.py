"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.7 — Evidence Layer Unit Test Suite

Verifies all 14 required test cases:
TEST 1: Vision candidate becomes 'possible' rather than automatically 'supported'.
TEST 2: Explicit QA answer 'yes' produces appropriate 'supported' status.
TEST 3: Explicit QA answer 'no' produces 'absent' status.
TEST 4: QA answer 'uncertain' remains 'uncertain' (or 'possible' if backed by vision score).
TEST 5: Missing answer remains 'uncertain' / 'possible' without fabricating 'yes'.
TEST 6: Model score is preserved as 'model_score'.
TEST 7: No 'confidence' field is generated anywhere in the output.
TEST 8: Location remains 'unspecified' when no evidence exists.
TEST 9: Severity remains 'unspecified' when no evidence exists.
TEST 10: Multiple evidence sources are preserved and tracked.
TEST 11: Ground truth does NOT appear in production LLM input package.
TEST 12: Invalid model scores (<0 or >1 or non-numeric) are rejected.
TEST 13: Duplicate findings are deduplicated cleanly.
TEST 14: Evidence package passes validate_evidence_package validation.
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

from evidence_layer import EvidenceLayer
from diagnostic_qa import DiagnosticQAEngine
from validate_evidence import validate_evidence_package


class TestPhase07EvidenceLayer(unittest.TestCase):

    def setUp(self):
        self.evidence_layer = EvidenceLayer()
        self.qa_engine = DiagnosticQAEngine()
        self.mock_scores = {
            "Pneumothorax": 0.25,
            "Effusion": 0.72,
            "Consolidation": 0.15,
            "Cardiomegaly": 0.05,
            "Atelectasis": 0.85
        }

    def test_01_vision_candidate_becomes_possible(self):
        """TEST 1: Vision candidate without confirmed QA answer becomes 'possible'."""
        qa_output = self.qa_engine.evaluate_study(
            study_id="TEST_01",
            image_id="IMG_01",
            view="Frontal",
            pathology_scores=self.mock_scores
        )
        pkg = self.evidence_layer.build_evidence_package(qa_output)
        atel_finding = next(f for f in pkg["findings"] if f["finding"] == "Atelectasis")
        # In prototype mode where QA presence is uncertain, vision activation makes it 'possible'
        self.assertEqual(atel_finding["status"], "possible")
        self.assertNotEqual(atel_finding["status"], "supported")

    def test_02_explicit_qa_answer_yes_produces_supported(self):
        """TEST 2: Explicit QA answer 'yes' produces 'supported' status."""
        user_answers = {
            "Effusion": {"presence": "yes", "location": "right", "severity": "small"}
        }
        qa_output = self.qa_engine.evaluate_study(
            study_id="TEST_02",
            image_id="IMG_02",
            view="Frontal",
            pathology_scores=self.mock_scores,
            supplied_answers=user_answers
        )
        pkg = self.evidence_layer.build_evidence_package(qa_output)
        eff_finding = next(f for f in pkg["findings"] if f["finding"] == "Effusion")
        self.assertEqual(eff_finding["status"], "supported")
        self.assertEqual(eff_finding["location"], "right")
        self.assertEqual(eff_finding["severity"], "small")

    def test_03_explicit_qa_answer_no_produces_absent(self):
        """TEST 3: Explicit QA answer 'no' produces 'absent' status."""
        user_answers = {
            "Pneumothorax": {"presence": "no"}
        }
        qa_output = self.qa_engine.evaluate_study(
            study_id="TEST_03",
            image_id="IMG_03",
            view="Frontal",
            pathology_scores=self.mock_scores,
            supplied_answers=user_answers
        )
        pkg = self.evidence_layer.build_evidence_package(qa_output)
        pneu_finding = next(f for f in pkg["findings"] if f["finding"] == "Pneumothorax")
        self.assertEqual(pneu_finding["status"], "absent")

    def test_04_qa_answer_uncertain_handled(self):
        """TEST 4: QA answer 'uncertain' produces 'possible' with vision score or 'uncertain'."""
        user_answers = {
            "Consolidation": {"presence": "uncertain"}
        }
        qa_output = self.qa_engine.evaluate_study(
            study_id="TEST_04",
            image_id="IMG_04",
            view="Frontal",
            pathology_scores=self.mock_scores,
            supplied_answers=user_answers
        )
        pkg = self.evidence_layer.build_evidence_package(qa_output)
        cons_finding = next(f for f in pkg["findings"] if f["finding"] == "Consolidation")
        self.assertIn(cons_finding["status"], ["possible", "uncertain"])
        self.assertNotEqual(cons_finding["status"], "supported")

    def test_05_missing_answer_remains_uncertain_or_possible(self):
        """TEST 5: Missing answer does NOT silently become 'yes' or 'supported'."""
        qa_output = self.qa_engine.evaluate_study(
            study_id="TEST_05",
            image_id="IMG_05",
            view="Frontal",
            pathology_scores=self.mock_scores
        )
        pkg = self.evidence_layer.build_evidence_package(qa_output)
        for f in pkg["findings"]:
            self.assertNotEqual(f["status"], "supported", f"Unanswered finding {f['finding']} became supported!")

    def test_06_model_score_preserved(self):
        """TEST 6: Model score is preserved as numeric 'model_score'."""
        qa_output = self.qa_engine.evaluate_study(
            study_id="TEST_06",
            image_id="IMG_06",
            view="Frontal",
            pathology_scores=self.mock_scores
        )
        pkg = self.evidence_layer.build_evidence_package(qa_output)
        for f in pkg["findings"]:
            self.assertIn("model_score", f)
            self.assertIsInstance(f["model_score"], float)

    def test_07_no_confidence_field_generated(self):
        """TEST 7: No 'confidence' field is present in Evidence Layer or LLM Input."""
        qa_output = self.qa_engine.evaluate_study(
            study_id="TEST_07",
            image_id="IMG_07",
            view="Frontal",
            pathology_scores=self.mock_scores
        )
        pkg = self.evidence_layer.build_evidence_package(qa_output)
        pkg_str = json.dumps(pkg)
        self.assertNotIn('"confidence"', pkg_str)

    def test_08_location_unspecified_when_no_evidence(self):
        """TEST 8: Location defaults to 'unspecified' without evidence."""
        qa_output = self.qa_engine.evaluate_study(
            study_id="TEST_08",
            image_id="IMG_08",
            view="Frontal",
            pathology_scores=self.mock_scores
        )
        pkg = self.evidence_layer.build_evidence_package(qa_output)
        for f in pkg["findings"]:
            self.assertEqual(f["location"], "unspecified")

    def test_09_severity_unspecified_when_no_evidence(self):
        """TEST 9: Severity defaults to 'unspecified' without evidence."""
        qa_output = self.qa_engine.evaluate_study(
            study_id="TEST_09",
            image_id="IMG_09",
            view="Frontal",
            pathology_scores=self.mock_scores
        )
        pkg = self.evidence_layer.build_evidence_package(qa_output)
        for f in pkg["findings"]:
            self.assertEqual(f["severity"], "unspecified")

    def test_10_multiple_evidence_sources_preserved(self):
        """TEST 10: Multiple evidence sources are tracked in provenance."""
        user_answers = {
            "Atelectasis": {"presence": "yes", "location": "basilar"}
        }
        qa_output = self.qa_engine.evaluate_study(
            study_id="TEST_10",
            image_id="IMG_10",
            view="Frontal",
            pathology_scores=self.mock_scores,
            supplied_answers=user_answers
        )
        pkg = self.evidence_layer.build_evidence_package(qa_output)
        atel_finding = next(f for f in pkg["findings"] if f["finding"] == "Atelectasis")
        self.assertIn("vision_model", atel_finding["evidence_sources"])
        self.assertIn("diagnostic_qa", atel_finding["evidence_sources"])

    def test_11_ground_truth_absent_from_production_llm_input(self):
        """TEST 11: Ground truth data does NOT leak into production LLM Input Package."""
        mock_gt = {
            "study_id": "TEST_11",
            "ground_truth_findings": [{"finding": "Pneumothorax", "presence": "no"}]
        }
        qa_output = self.qa_engine.evaluate_study(
            study_id="TEST_11",
            image_id="IMG_11",
            view="Frontal",
            pathology_scores=self.mock_scores
        )
        # Build production package with ground_truth passed
        pkg = self.evidence_layer.build_evidence_package(qa_output, ground_truth=mock_gt, is_evaluation_mode=False)
        llm_pkg_str = json.dumps(pkg["llm_input_package"])
        self.assertNotIn("report_ground_truth", llm_pkg_str)
        self.assertNotIn("ground_truth_findings", llm_pkg_str)

    def test_12_invalid_model_scores_rejected(self):
        """TEST 12: Invalid model scores (<0, >1, non-numeric) raise ValueError."""
        with self.assertRaises(ValueError):
            self.evidence_layer.validate_score(-0.5)
        with self.assertRaises(ValueError):
            self.evidence_layer.validate_score(1.5)
        with self.assertRaises(ValueError):
            self.evidence_layer.validate_score("not_a_number")

    def test_13_duplicate_findings_handled(self):
        """TEST 13: Findings are cleanly deduplicated."""
        raw_qa = {
            "study_id": "TEST_13",
            "image_id": "IMG_13",
            "view": "Frontal",
            "vision_candidates": [
                {"finding": "Effusion", "model_score": 0.8},
                {"finding": "Effusion", "model_score": 0.8}
            ],
            "qa_findings": [
                {"finding": "Effusion", "presence": "yes", "model_score": 0.8}
            ]
        }
        pkg = self.evidence_layer.build_evidence_package(raw_qa)
        effusion_entries = [f for f in pkg["findings"] if f["finding"] == "Effusion"]
        self.assertEqual(len(effusion_entries), 1)

    def test_14_evidence_json_schema_validation(self):
        """TEST 14: Complete evidence package passes validate_evidence_package checks."""
        qa_output = self.qa_engine.evaluate_study(
            study_id="TEST_14",
            image_id="IMG_14",
            view="Frontal",
            pathology_scores=self.mock_scores
        )
        pkg = self.evidence_layer.build_evidence_package(qa_output)
        is_valid, errors = validate_evidence_package(pkg)
        self.assertTrue(is_valid, f"Evidence package validation failed with errors: {errors}")


if __name__ == "__main__":
    unittest.main()
