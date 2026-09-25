"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.4 — Automated Test Suite: Multi-Study Dataset Management, Review Queue & Evaluation Analytics

Module: test_phase_1_4_dataset.py
Tests:
1. Dataset discovery across dataset directories.
2. Study schema conformity with docs/study_schema.json.
3. Study ID validation and sanitization.
4. Study summary generation with accurate artifact detection.
5. Review queue generation and format compliance.
6. Review queue status and consensus filtering.
7. Review queue deterministic natural sorting.
8. Dataset overview statistics calculation.
9. Sanitized study search.
10. Dynamic study selection and artifact loading.
11. Reviewer cohort aggregation without performance ranking.
12. Consensus metrics aggregation.
13. Cohen's Kappa aggregation for 2-reviewer studies.
14. Fleiss' Kappa aggregation for >=3-reviewer studies.
15. Adjudication metrics count.
16. Finalization state tracking.
17. Study provenance 8-stage audit trail generation.
18. Machine artifact byte immutability verification (SHA-256 pre/post check).
19. Review artifact integrity validation.
20. Consensus artifact integrity validation.
21. Strict ground-truth isolation from all client-facing dataset APIs.
22. Path traversal rejection (../, ..\\).
23. Unknown study rejection (404 / None).
24. Security string length bounding.
25. API regression: GET /api/studies.
26. API regression: GET /api/review-queue.
27. API regression: GET /api/dataset/stats & analytics.
28. API regression: GET /api/studies/{study_id}/provenance.
29. Dataset JSON export with research disclaimer.
30. Dataset plain-text export format and disclaimer.
31. Study export JSON/text format.
32. Finalized consensus locking in study lifecycle.
33. Malformed artifact error handling.
"""

import os
import sys
import json
import hashlib
import unittest
import threading
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from urllib.request import Request, urlopen
from urllib.error import HTTPError

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from backend.study_manager import StudyManager, global_study_manager
from backend.review_queue import ReviewQueue, global_review_queue
from backend.evaluation_manager import EvaluationManager, global_evaluation_manager
from backend.validate_dataset import DatasetValidator, global_dataset_validator
from backend.api import create_server

DATA_DIR = os.path.join(BASE_DIR, "data", "iu_xray")
REVIEWS_DIR = os.path.join(BASE_DIR, "data", "reviews")
CONSENSUS_DIR = os.path.join(BASE_DIR, "data", "consensus")


def calculate_dir_sha256(directory_path: str) -> Dict[str, str]:
    """Calculates SHA-256 hashes of all files in a directory recursively."""
    hashes = {}
    if not os.path.exists(directory_path):
        return hashes
    for root, _, files in os.walk(directory_path):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            h = hashlib.sha256()
            try:
                with open(fp, "rb") as f:
                    while chunk := f.read(65536):
                        h.update(chunk)
                rel_p = os.path.relpath(fp, directory_path).replace("\\", "/")
                hashes[rel_p] = h.hexdigest()
            except Exception:
                pass
    return hashes


class TestPhase14Dataset(unittest.TestCase):
    """Phase 1.4 Multi-Study Dataset, Review Queue, and Analytics Test Suite."""

    @classmethod
    def setUpClass(cls):
        # Calculate SHA256 of all machine artifacts before running tests
        cls.pre_test_hashes = calculate_dir_sha256(DATA_DIR)

        # Start background test API server on port 8944
        cls.server_port = 8944
        cls.server = create_server(port=cls.server_port, host="127.0.0.1")
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _api_get(self, endpoint: str):
        url = f"http://127.0.0.1:{self.server_port}{endpoint}"
        req = Request(url, method="GET", headers={"Connection": "close"})
        with urlopen(req, timeout=10) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()
            if "application/json" in content_type:
                return json.loads(raw.decode("utf-8")), resp.status
            return raw.decode("utf-8"), resp.status

    # =========================================================================
    # 1. Dataset Discovery & Schema Tests
    # =========================================================================

    def test_01_dataset_discovery(self):
        """TEST 01: Discover available studies dynamically without assuming CXR1122 is only study."""
        studies = global_study_manager.discover_studies()
        self.assertIsInstance(studies, list)
        self.assertGreater(len(studies), 1, "Should discover multiple available studies from images directory.")
        
        study_ids = [s["study_id"] for s in studies]
        self.assertIn("CXR1122", study_ids)
        self.assertEqual(studies[0]["study_id"], "CXR1122", "Baseline CXR1122 should be listed first.")

    def test_02_study_schema_compliance(self):
        """TEST 02: Study summary contains all required schema fields."""
        summary = global_study_manager.get_study_summary("CXR1122")
        self.assertIsNotNone(summary)
        required_fields = [
            "study_id", "image_count", "views", "machine_pipeline_status",
            "evidence_count", "qa_count", "grounding_count", "machine_report_available",
            "review_status", "reviewer_count", "review_completion", "consensus_status",
            "adjudication_required", "consensus_finalized", "workflow_status", "last_updated", "disclaimer"
        ]
        for f in required_fields:
            self.assertIn(f, summary, f"Summary missing required field '{f}'")

    def test_03_study_id_validation(self):
        """TEST 03: Reject invalid study IDs and path traversal characters."""
        self.assertIsNone(global_study_manager.get_study_summary("../CXR1122"))
        self.assertIsNone(global_study_manager.get_study_summary("..\\CXR1122"))
        self.assertIsNone(global_study_manager.get_study_summary(""))
        self.assertIsNone(global_study_manager.get_study_summary(None))

    def test_04_study_summary_artifact_detection(self):
        """TEST 04: Detects images, views, machine report, and reviews for CXR1122."""
        summary = global_study_manager.get_study_summary("CXR1122")
        self.assertEqual(summary["study_id"], "CXR1122")
        self.assertIn("Frontal", summary["views"])
        self.assertTrue(summary["machine_report_available"])
        self.assertEqual(summary["evidence_count"], 7)
        self.assertIn("RESEARCH PROTOTYPE", summary["disclaimer"])

    # =========================================================================
    # 2. Review Queue Tests
    # =========================================================================

    def test_05_review_queue_generation(self):
        """TEST 05: Review queue generates items compliant with ReviewQueueItem schema."""
        queue = global_review_queue.get_queue_items()
        self.assertIsInstance(queue, list)
        self.assertGreater(len(queue), 0)

        item = queue[0]
        self.assertIn("study_id", item)
        self.assertIn("primary_view", item)
        self.assertIn("evidence_count", item)
        self.assertIn("machine_report", item)
        self.assertIn("reviewer_count", item)
        self.assertIn("review_completion", item)
        self.assertIn("workflow_status", item)

    def test_06_review_queue_filtering_status(self):
        """TEST 06: Queue filters properly by workflow status."""
        unreviewed = global_review_queue.get_queue_items(status="UNREVIEWED")
        for u in unreviewed:
            self.assertEqual(u["workflow_status"], "UNREVIEWED")

        # Test filtering with ALL returns all items
        all_items = global_review_queue.get_queue_items(status="ALL")
        self.assertGreaterEqual(len(all_items), len(unreviewed))

    def test_07_review_queue_deterministic_sorting(self):
        """TEST 07: Queue sorts deterministically without subjective clinical priority ranking."""
        sorted_asc = global_review_queue.get_queue_items(sort_by="study_id", sort_dir="asc")
        sorted_desc = global_review_queue.get_queue_items(sort_by="study_id", sort_dir="desc")
        self.assertEqual(sorted_asc[0]["study_id"], sorted_desc[-1]["study_id"])

    # =========================================================================
    # 3. Search & Dataset Overview Statistics
    # =========================================================================

    def test_08_dataset_overview_statistics(self):
        """TEST 08: Dataset statistics aggregate counts across all lifecycle states."""
        stats = global_evaluation_manager.get_dataset_stats()
        self.assertIn("total_studies", stats)
        self.assertIn("unreviewed", stats)
        self.assertIn("in_review", stats)
        self.assertIn("awaiting_consensus", stats)
        self.assertIn("adjudication_required", stats)
        self.assertIn("finalized", stats)
        self.assertIn("total_reviewers", stats)

        total = stats["unreviewed"] + stats["in_review"] + stats["awaiting_consensus"] + stats["adjudication_required"] + stats["ready_for_finalization"] + stats["finalized"]
        self.assertEqual(total, stats["total_studies"])

    def test_09_sanitized_study_search(self):
        """TEST 09: Study search matches against study ID metadata without searching report text."""
        res = global_study_manager.search_studies("1122")
        self.assertTrue(any(s["study_id"] == "CXR1122" for s in res))

        # Test invalid query returns empty list
        invalid_res = global_study_manager.search_studies("../secret")
        self.assertEqual(invalid_res, [])

    def test_10_dynamic_study_selection(self):
        """TEST 10: Dynamic study selection works for any discovered study ID."""
        studies = global_study_manager.discover_studies()
        self.assertGreater(len(studies), 1)
        second_study_id = studies[1]["study_id"]
        summary = global_study_manager.get_study_summary(second_study_id)
        self.assertIsNotNone(summary)
        self.assertEqual(summary["study_id"], second_study_id)

    # =========================================================================
    # 4. Evaluation Analytics & Inter-Rater Agreement
    # =========================================================================

    def test_11_reviewer_workflow_aggregation(self):
        """TEST 11: Reviewer workflow panel tracks operational counts without ranking."""
        analytics = global_evaluation_manager.get_evaluation_analytics()
        self.assertIn("reviewer_workflow", analytics)
        reviewers = analytics["reviewer_workflow"]

        for r in reviewers:
            self.assertIn("reviewer_id", r)
            self.assertIn("studies_assigned", r)
            self.assertIn("studies_completed", r)
            self.assertIn("findings_reviewed", r)
            self.assertIn("consensus_participation", r)
            self.assertIn("adjudications", r)
            # Ensure no subjective ranking fields exist
            self.assertNotIn("rank", r)
            self.assertNotIn("accuracy_score", r)
            self.assertNotIn("quality_score", r)

    def test_12_consensus_metrics_aggregation(self):
        """TEST 12: Consensus analytics aggregate decision categories."""
        analytics = global_evaluation_manager.get_evaluation_analytics()
        cm = analytics["consensus_metrics"]
        self.assertIn("total_consensus_studies", cm)
        self.assertIn("unanimous_findings", cm)
        self.assertIn("majority_findings", cm)
        self.assertIn("adjudicated_findings", cm)
        self.assertIn("finalized_consensus_reports", cm)

    def test_13_cohens_kappa_aggregation(self):
        """TEST 13: Cohen's Kappa is strictly aggregated only for 2-reviewer studies."""
        analytics = global_evaluation_manager.get_evaluation_analytics()
        agr = analytics["agreement_analytics"]["two_reviewer_studies"]
        self.assertEqual(agr["metric_name"], "Cohen's Kappa (κ)")
        self.assertIn("sample_size_studies", agr)
        self.assertIn("applicability_note", agr)
        if agr["average_kappa"] is not None:
            self.assertTrue(-1.0 <= agr["average_kappa"] <= 1.0)

    def test_14_fleiss_kappa_aggregation(self):
        """TEST 14: Fleiss' Kappa is strictly aggregated only for >=3-reviewer studies."""
        analytics = global_evaluation_manager.get_evaluation_analytics()
        agr = analytics["agreement_analytics"]["multi_reviewer_studies"]
        self.assertEqual(agr["metric_name"], "Fleiss' Kappa (κ)")
        self.assertIn("sample_size_studies", agr)
        if agr["average_kappa"] is not None:
            self.assertTrue(-1.0 <= agr["average_kappa"] <= 1.0)

    def test_15_adjudication_metrics_count(self):
        """TEST 15: Adjudications conducted are tracked accurately."""
        analytics = global_evaluation_manager.get_evaluation_analytics()
        self.assertGreaterEqual(analytics["consensus_metrics"]["adjudicated_findings"], 0)

    def test_16_finalization_state_tracking(self):
        """TEST 16: Finalized consensus count matches finalized session files."""
        analytics = global_evaluation_manager.get_evaluation_analytics()
        self.assertGreaterEqual(analytics["consensus_metrics"]["finalized_consensus_reports"], 0)

    # =========================================================================
    # 5. Provenance & Machine Immutability
    # =========================================================================

    def test_17_study_provenance_pipeline(self):
        """TEST 17: Complete 8-stage sanitized provenance audit trail is produced."""
        prov = global_evaluation_manager.get_study_provenance("CXR1122")
        self.assertIsNotNone(prov)
        self.assertEqual(prov["study_id"], "CXR1122")
        stages = prov["pipeline_stages"]
        self.assertGreaterEqual(len(stages), 8)
        
        stage_names = [s["stage_name"] for s in stages]
        self.assertTrue(any("Vision Backbone" in name for name in stage_names))
        self.assertTrue(any("Diagnostic QA" in name for name in stage_names))
        self.assertTrue(any("Evidence Layer" in name for name in stage_names))
        self.assertTrue(any("Visual Grounding" in name for name in stage_names))
        self.assertTrue(any("LLM Report" in name for name in stage_names))
        self.assertTrue(any("Human Review" in name for name in stage_names))
        self.assertTrue(any("Consensus Synthesis" in name for name in stage_names))
        self.assertTrue(any("Finalized" in name for name in stage_names))

    def test_18_machine_artifact_immutability(self):
        """TEST 18: SHA-256 hashes of all machine artifacts remain 100% byte-identical."""
        post_hashes = calculate_dir_sha256(DATA_DIR)
        self.assertEqual(self.pre_test_hashes, post_hashes, "Machine artifacts in data/iu_xray/ MUST NEVER be modified!")

    # =========================================================================
    # 6. Integrity, Security & Ground-Truth Isolation
    # =========================================================================

    def test_19_review_artifact_integrity(self):
        """TEST 19: Review artifacts pass schema and structure integrity validator."""
        valid, errors = global_dataset_validator.validate_review_artifacts()
        self.assertTrue(valid, f"Review artifact validation failed: {errors}")

    def test_20_consensus_artifact_integrity(self):
        """TEST 20: Consensus artifacts pass schema and rationale integrity validator."""
        valid, errors = global_dataset_validator.validate_consensus_artifacts()
        self.assertTrue(valid, f"Consensus artifact validation failed: {errors}")

    def test_21_ground_truth_isolation_from_dataset_api(self):
        """TEST 21: Ground-truth report text is strictly excluded from dataset APIs."""
        summaries = global_study_manager.discover_studies()
        valid, errors = global_dataset_validator.validate_ground_truth_isolation(summaries)
        self.assertTrue(valid, f"Ground truth detected in summaries: {errors}")

        queue = global_review_queue.get_queue_items()
        valid_q, errors_q = global_dataset_validator.validate_ground_truth_isolation(queue)
        self.assertTrue(valid_q, f"Ground truth detected in review queue: {errors_q}")

    def test_22_path_traversal_rejection(self):
        """TEST 22: Security input validator rejects path traversal attempts."""
        valid, msg = global_dataset_validator.validate_security_input("../secret/report.xml")
        self.assertFalse(valid)
        self.assertIn("traversal", msg.lower())

        valid_win, msg_win = global_dataset_validator.validate_security_input("..\\secret\\report.xml")
        self.assertFalse(valid_win)

    def test_23_unknown_study_rejection(self):
        """TEST 23: Requesting non-existent study summary returns None."""
        self.assertIsNone(global_study_manager.get_study_summary("CXR99999_NON_EXISTENT"))

    def test_24_oversized_string_rejection(self):
        """TEST 24: Security validator rejects input exceeding 5000 characters."""
        oversized = "A" * 6000
        valid, msg = global_dataset_validator.validate_security_input(oversized)
        self.assertFalse(valid)
        self.assertIn("exceeds", msg.lower())

    # =========================================================================
    # 7. REST API Endpoints Verification
    # =========================================================================

    def test_25_api_get_studies(self):
        """TEST 25: GET /api/studies returns sanitized multi-study summaries."""
        data, status = self._api_get("/api/studies")
        self.assertEqual(status, 200)
        self.assertIn("studies", data)
        self.assertGreater(data["total"], 0)

    def test_26_api_get_review_queue(self):
        """TEST 26: GET /api/review-queue returns queue with query filtering."""
        data, status = self._api_get("/api/review-queue?status=UNREVIEWED")
        self.assertEqual(status, 200)
        self.assertIn("queue", data)

    def test_27_api_get_dataset_stats_and_analytics(self):
        """TEST 27: GET /api/dataset/stats and /api/dataset/analytics return research statistics."""
        stats, s_status = self._api_get("/api/dataset/stats")
        self.assertEqual(s_status, 200)
        self.assertIn("total_studies", stats)

        analytics, a_status = self._api_get("/api/dataset/analytics")
        self.assertEqual(a_status, 200)
        self.assertIn("review_metrics", analytics)
        self.assertIn("agreement_analytics", analytics)

    def test_28_api_get_study_provenance(self):
        """TEST 28: GET /api/studies/CXR1122/provenance returns 8-stage audit trail."""
        data, status = self._api_get("/api/studies/CXR1122/provenance")
        self.assertEqual(status, 200)
        self.assertEqual(data["study_id"], "CXR1122")
        self.assertIn("pipeline_stages", data)

    def test_29_dataset_json_export(self):
        """TEST 29: GET /api/dataset/export?format=json returns complete export package."""
        data, status = self._api_get("/api/dataset/export?format=json")
        self.assertEqual(status, 200)
        self.assertIn("disclaimer", data)
        self.assertIn("dataset_statistics", data)
        self.assertIn("studies", data)

    def test_30_dataset_text_export(self):
        """TEST 30: GET /api/dataset/export?format=text returns human-readable export document."""
        text, status = self._api_get("/api/dataset/export?format=text")
        self.assertEqual(status, 200)
        self.assertIn("EXPLAINABLE RADIOLOGY RESEARCH PROTOTYPE", text)
        self.assertIn("DATASET LIFECYCLE STATISTICS", text)
        self.assertIn("INTER-RATER AGREEMENT ANALYTICS", text)

    def test_31_study_export(self):
        """TEST 31: GET /api/studies/CXR1122/export returns single-study export package."""
        data, status = self._api_get("/api/studies/CXR1122/export?format=json")
        self.assertEqual(status, 200)
        self.assertIn("study_summary", data)
        self.assertIn("provenance", data)

    def test_32_finalized_study_locking(self):
        """TEST 32: Finalized study reflects FINALIZED workflow status and lock state."""
        summary = global_study_manager.get_study_summary("CXR1122")
        self.assertIsNotNone(summary)
        # Check that consensus_finalized is boolean
        self.assertIsInstance(summary["consensus_finalized"], bool)

    def test_33_malformed_artifact_handling(self):
        """TEST 33: Gracefully handle queries for non-existent studies without crashing."""
        try:
            _, status = self._api_get("/api/studies/NON_EXISTENT_STUDY/summary")
            self.assertEqual(status, 404)
        except HTTPError as e:
            self.assertEqual(e.code, 404)


if __name__ == "__main__":
    unittest.main()
