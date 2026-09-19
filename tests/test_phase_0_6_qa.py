"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.6 — Unit Test Suite

Tests all 13 required verification cases for Diagnostic QA Engine and Report Parsing:
TEST 1: Baseline findings are always selected.
TEST 2: Top-K non-baseline findings are selected correctly.
TEST 3: Threshold filtering works.
TEST 4: TOP_K and QA_THRESHOLD can be configured.
TEST 5: No duplicate findings are generated.
TEST 6: Conditional location questions are only created when appropriate.
TEST 7: Severity remains 'unspecified' when unsupported.
TEST 8: Vision model score is stored as 'model_score', not 'confidence'.
TEST 9: XML report parsing works.
TEST 10: Image/report study IDs remain correctly associated.
TEST 11: Normal report example can be converted into structured findings.
TEST 12: Abnormal report example can be converted into structured findings.
TEST 13: Multiple-finding report can be converted into structured findings.
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

from diagnostic_qa import DiagnosticQAEngine, DEFAULT_BASELINE_FINDINGS
from report_parser import parse_iu_report
from evaluate_qa import evaluate_qa_against_ground_truth


class TestPhase06DiagnosticQA(unittest.TestCase):

    def setUp(self):
        self.engine = DiagnosticQAEngine(top_k=3, qa_threshold=0.35)
        # Mock 18 pathology activation scores
        self.mock_scores = {
            "Pneumothorax": 0.05,
            "Effusion": 0.08,
            "Consolidation": 0.12,
            "Cardiomegaly": 0.02,
            "Atelectasis": 0.72,
            "Nodule": 0.65,
            "Infiltration": 0.45,
            "Fracture": 0.38,
            "Edema": 0.10,
            "Emphysema": 0.05,
            "Fibrosis": 0.02,
            "Pneumonia": 0.01,
            "Pleural_Thickening": 0.01,
            "Mass": 0.03,
            "Hernia": 0.01,
            "Lung Lesion": 0.01,
            "Lung Opacity": 0.15,
            "Enlarged Cardiomediastinum": 0.01
        }
        self.reports_dir = os.path.join(BASE_DIR, "data", "iu_xray", "reports", "ecgen-radiology")

    def test_01_baseline_findings_always_selected(self):
        """TEST 1: Baseline findings are always selected regardless of scores."""
        candidates = self.engine.select_candidates(self.mock_scores)
        candidate_names = [c["finding"] for c in candidates]
        for base in DEFAULT_BASELINE_FINDINGS:
            self.assertIn(base, candidate_names, f"Baseline finding {base} was not selected.")

    def test_02_top_k_non_baseline_selected_correctly(self):
        """TEST 2: Top-K non-baseline findings are selected in descending score order."""
        # Top non-baseline in mock_scores: Atelectasis (0.72), Nodule (0.65), Infiltration (0.45)
        candidates = self.engine.select_candidates(self.mock_scores, top_k=3, qa_threshold=0.90)
        candidate_names = [c["finding"] for c in candidates]
        self.assertIn("Atelectasis", candidate_names)
        self.assertIn("Nodule", candidate_names)
        self.assertIn("Infiltration", candidate_names)
        # Fracture (0.38) is #4 and below threshold 0.90, so should NOT be included
        self.assertNotIn("Fracture", candidate_names)

    def test_03_threshold_filtering_works(self):
        """TEST 3: Non-baseline findings above threshold are included even if beyond Top-K."""
        # Top-1 is Atelectasis (0.72). Nodule (0.65), Infiltration (0.45), Fracture (0.38) all >= 0.35
        candidates = self.engine.select_candidates(self.mock_scores, top_k=1, qa_threshold=0.35)
        candidate_names = [c["finding"] for c in candidates]
        self.assertIn("Atelectasis", candidate_names)
        self.assertIn("Nodule", candidate_names)
        self.assertIn("Infiltration", candidate_names)
        self.assertIn("Fracture", candidate_names)
        # Edema (0.10) is < 0.35, so should NOT be included
        self.assertNotIn("Edema", candidate_names)

    def test_04_top_k_and_threshold_configurable(self):
        """TEST 4: TOP_K and QA_THRESHOLD can be dynamically configured."""
        custom_engine = DiagnosticQAEngine(top_k=1, qa_threshold=0.80)
        candidates = custom_engine.select_candidates(self.mock_scores)
        non_baseline = [c for c in candidates if not c["is_baseline"]]
        # Only Atelectasis (0.72) is Top-1, no items >= 0.80
        self.assertEqual(len(non_baseline), 1)
        self.assertEqual(non_baseline[0]["finding"], "Atelectasis")

    def test_05_no_duplicate_findings(self):
        """TEST 5: Output candidates contain no duplicate findings."""
        candidates = self.engine.select_candidates(self.mock_scores, top_k=5, qa_threshold=0.01)
        candidate_names = [c["finding"] for c in candidates]
        self.assertEqual(len(candidate_names), len(set(candidate_names)), "Duplicate findings detected.")

    def test_06_conditional_location_questions(self):
        """TEST 6: Location questions are created only when appropriate."""
        # When presence is 'no', level 2 location questions must NOT be evaluated
        user_answers = {
            "Pneumothorax": {"presence": "no"},
            "Atelectasis": {"presence": "yes", "location": "basilar"}
        }
        output = self.engine.evaluate_study(
            study_id="TEST_01",
            image_id="IMG_01",
            view="Frontal",
            pathology_scores=self.mock_scores,
            supplied_answers=user_answers
        )
        questions = output["questions_evaluated"]
        pneu_questions = [q for q in questions if q["finding"] == "Pneumothorax"]
        atel_questions = [q for q in questions if q["finding"] == "Atelectasis"]

        # Pneumothorax answered 'no' -> only Level 1 evaluated
        self.assertEqual(len(pneu_questions), 1)
        self.assertEqual(pneu_questions[0]["level"], 1)

        # Atelectasis answered 'yes' -> Level 1 and Level 2 evaluated
        atel_levels = [q["level"] for q in atel_questions]
        self.assertIn(1, atel_levels)
        self.assertIn(2, atel_levels)

    def test_07_severity_remains_unspecified_when_unsupported(self):
        """TEST 7: Severity remains 'unspecified' and is not hallucinated."""
        user_answers = {
            "Atelectasis": {"presence": "yes", "location": "basilar"}
            # No severity supplied
        }
        output = self.engine.evaluate_study(
            study_id="TEST_02",
            image_id="IMG_02",
            view="Frontal",
            pathology_scores=self.mock_scores,
            supplied_answers=user_answers
        )
        atel_finding = next(f for f in output["qa_findings"] if f["finding"] == "Atelectasis")
        self.assertEqual(atel_finding["severity"], "unspecified")
        self.assertEqual(atel_finding["location"], "basilar")

    def test_08_model_score_naming_convention(self):
        """TEST 8: Vision model score is stored as 'model_score', never 'confidence'."""
        output = self.engine.evaluate_study(
            study_id="TEST_03",
            image_id="IMG_03",
            view="Frontal",
            pathology_scores=self.mock_scores
        )
        # Check vision candidates
        for cand in output["vision_candidates"]:
            self.assertIn("model_score", cand)
            self.assertNotIn("confidence", cand)
        # Check findings
        for finding in output["qa_findings"]:
            self.assertIn("model_score", finding)
            self.assertNotIn("confidence", finding)

    def test_09_xml_report_parsing(self):
        """TEST 9: XML report parsing extracts sections and ground truth accurately."""
        sample_xml = os.path.join(self.reports_dir, "1122.xml")
        if os.path.exists(sample_xml):
            report_data = parse_iu_report(sample_xml)
            self.assertEqual(report_data["study_id"], "CXR1122")
            self.assertIn("findings", report_data["sections"])
            self.assertIn("impression", report_data["sections"])
            self.assertTrue(len(report_data["parent_images"]) > 0)
            self.assertTrue(len(report_data["ground_truth_findings"]) > 0)

    def test_10_image_report_pairing_association(self):
        """TEST 10: Study ID and Image ID remain correctly paired in parsed reports."""
        sample_xml = os.path.join(self.reports_dir, "1122.xml")
        if os.path.exists(sample_xml):
            report_data = parse_iu_report(sample_xml)
            parent_ids = [img["image_id"] for img in report_data["parent_images"]]
            self.assertIn("CXR1122_IM-0080-1001-0002", parent_ids)

    def test_11_normal_report_conversion(self):
        """TEST 11: Normal report example converts into negative ground-truth findings."""
        sample_xml = os.path.join(self.reports_dir, "1.xml")
        if os.path.exists(sample_xml):
            report_data = parse_iu_report(sample_xml)
            gt_findings = {f["finding"]: f["presence"] for f in report_data["ground_truth_findings"]}
            self.assertEqual(gt_findings.get("Pneumothorax"), "no")
            self.assertEqual(gt_findings.get("Effusion"), "no")
            self.assertEqual(gt_findings.get("Consolidation"), "no")

    def test_12_abnormal_report_conversion(self):
        """TEST 12: Single abnormal finding is extracted as positive with location/severity."""
        sample_xml = os.path.join(self.reports_dir, "1001.xml")
        if os.path.exists(sample_xml):
            report_data = parse_iu_report(sample_xml)
            fibrosis_f = next((f for f in report_data["ground_truth_findings"] if f["finding"] == "Fibrosis"), None)
            self.assertIsNotNone(fibrosis_f, "Fibrosis was not identified in CXR1001 report.")
            self.assertEqual(fibrosis_f["presence"], "yes")

    def test_13_multiple_finding_report_conversion(self):
        """TEST 13: Multiple abnormal findings are extracted correctly from complex report."""
        sample_xml = os.path.join(self.reports_dir, "1000.xml")
        if os.path.exists(sample_xml):
            report_data = parse_iu_report(sample_xml)
            gt_findings = {f["finding"]: f for f in report_data["ground_truth_findings"]}
            self.assertIn("Lung Opacity", gt_findings)
            self.assertIn("Atelectasis", gt_findings)
            self.assertIn(gt_findings["Lung Opacity"]["presence"], ["yes", "uncertain"])
            self.assertEqual(gt_findings["Pneumothorax"]["presence"], "no")
            self.assertEqual(gt_findings["Effusion"]["presence"], "no")


if __name__ == "__main__":
    unittest.main()
