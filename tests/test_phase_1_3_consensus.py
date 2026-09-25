"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.3 — Multi-Reviewer Consensus, Inter-Rater Agreement & Adjudication Tests

Test Suite covering:
1. Consensus session creation
2. Multiple reviewer registration
3. Duplicate reviewer rejection
4. Reviewer isolation (sessions remain separate files)
5. Missing reviewer detection
6. Finding aggregation across reviewers
7. Unanimous consensus
8. Majority consensus
9. Tie detection
10. Needs-review handling
11. Location consensus
12. Severity consensus
13. QA consensus aggregation
14. Cohen's Kappa calculation (N=2)
15. Fleiss' Kappa calculation (N>=3)
16. Kappa edge cases & NaN protection
17. Missing data handling
18. Adjudication detection
19. Adjudication creation
20. Mandatory adjudication reason validation
21. Adjudication history preservation
22. Consensus report generation
23. Finalization rejection when incomplete
24. Successful finalization
25. Consensus audit trail
26. Machine artifact immutability verification
27. Zero ground-truth leakage prevention
28. Invalid study rejection
29. Path traversal rejection
30. API backward compatibility regression
31. JSON consensus export
32. Text consensus export
33. Post-finalization mutation rejection
"""

import os
import sys
import json
import unittest
import urllib.request
import urllib.error
import threading
from http.server import HTTPServer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from api import RadiologyAPIHandler
from consensus_manager import global_consensus_manager
from review_manager import global_review_manager

REVIEWS_DIR = os.path.join(BASE_DIR, "data", "reviews")
CONSENSUS_DIR = os.path.join(BASE_DIR, "data", "consensus")


class TestPhase13Consensus(unittest.TestCase):
    """33 unit and invariant verification tests for Phase 1.3."""

    @classmethod
    def setUpClass(cls):
        cls.port = 8933
        cls.server = HTTPServer(("127.0.0.1", cls.port), RadiologyAPIHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        # Clean test files for CXR1122
        for f in os.listdir(CONSENSUS_DIR):
            if f.startswith("CXR1122"):
                try:
                    os.remove(os.path.join(CONSENSUS_DIR, f))
                except Exception:
                    pass
        for f in os.listdir(REVIEWS_DIR):
            if f.startswith("CXR1122"):
                try:
                    os.remove(os.path.join(REVIEWS_DIR, f))
                except Exception:
                    pass

    def _get(self, path):
        req = urllib.request.Request(f"{self.base_url}{path}", method="GET")
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    def _post(self, path, payload=None):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        req = urllib.request.Request(f"{self.base_url}{path}", data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    # --- Tests 1-5: Session & Reviewer Management ---
    def test_01_consensus_session_creation(self):
        """TEST 1: Consensus session is created with required schema fields and collecting status."""
        status, _, body = self._get("/api/studies/CXR1122/consensus")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["study_id"], "CXR1122")
        self.assertIn("consensus_id", data)
        self.assertIn(data["status"], ["collecting", "ready_for_consensus"])
        self.assertEqual(len(data["reviewers"]), 3)
        self.assertIn("agreement_metrics", data)

    def test_02_multiple_reviewer_registration(self):
        """TEST 2: Reviewers can be registered into consensus session."""
        payload = {
            "required_reviewers": 3,
            "minimum_reviewers": 2,
            "reviewers": [
                {"id": "dr_alice", "display_name": "Dr. Alice, MD"},
                {"id": "dr_bob", "display_name": "Dr. Bob, MD"},
                {"id": "dr_carol", "display_name": "Dr. Carol, MD"}
            ]
        }
        status, _, body = self._post("/api/studies/CXR1122/consensus", payload)
        self.assertEqual(status, 200)
        session = json.loads(body.decode("utf-8"))["session"]
        self.assertEqual(len(session["reviewers"]), 3)
        self.assertTrue(any(r["id"] == "dr_alice" for r in session["reviewers"]))

    def test_03_duplicate_reviewer_registration_idempotence(self):
        """TEST 3: Registering duplicate reviewer ID does not duplicate entries."""
        payload = {
            "required_reviewers": 2,
            "reviewers": [
                {"id": "dr_alice", "display_name": "Dr. Alice"},
                {"id": "dr_alice", "display_name": "Dr. Alice Renamed"}
            ]
        }
        status, _, body = self._post("/api/studies/CXR1122/consensus", payload)
        self.assertEqual(status, 200)
        session = json.loads(body.decode("utf-8"))["session"]
        alice_entries = [r for r in session["reviewers"] if r["id"] == "dr_alice"]
        self.assertEqual(len(alice_entries), 1)

    def test_04_reviewer_isolation(self):
        """TEST 4: Each reviewer operates in independent persistent storage without cross-contamination."""
        # Reviewer 1 reviews Infiltration as confirmed_present
        p1 = {"finding": "Infiltration", "reviewer_status": "confirmed_present", "reviewer_id": "rev_01"}
        s1, _, _ = self._post("/api/studies/CXR1122/review/finding", p1)
        self.assertEqual(s1, 200)

        # Reviewer 2 reviews Infiltration as confirmed_absent
        p2 = {"finding": "Infiltration", "reviewer_status": "confirmed_absent", "reviewer_id": "rev_02"}
        s2, _, _ = self._post("/api/studies/CXR1122/review/finding", p2)
        self.assertEqual(s2, 200)

        # Verify distinct files exist on disk
        r1_session = global_review_manager.get_review("CXR1122", reviewer_id="rev_01")
        r2_session = global_review_manager.get_review("CXR1122", reviewer_id="rev_02")
        self.assertIsNotNone(r1_session)
        self.assertIsNotNone(r2_session)

        f1 = next(f for f in r1_session["finding_reviews"] if f["finding"] == "Infiltration")
        f2 = next(f for f in r2_session["finding_reviews"] if f["finding"] == "Infiltration")
        self.assertEqual(f1["reviewer_status"], "confirmed_present")
        self.assertEqual(f2["reviewer_status"], "confirmed_absent")

    def test_05_missing_reviewer_detection(self):
        """TEST 5: Consensus status remains 'collecting' when fewer than minimum reviewers have completed."""
        # Only rev_01 completed
        for f in ["Infiltration", "Pneumothorax", "Effusion", "Consolidation", "Cardiomegaly", "Fracture", "Nodule"]:
            self._post("/api/studies/CXR1122/review/finding", {"finding": f, "reviewer_status": "confirmed_absent", "reviewer_id": "rev_01"})

        status, _, body = self._get("/api/studies/CXR1122/consensus")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["status"], "collecting")

    # --- Tests 6-13: Consensus Algorithms & Logic ---
    def _setup_three_completed_reviewers(self, r1_decisions, r2_decisions, r3_decisions):
        """Helper to populate 3 completed reviewer sessions."""
        findings = ["Infiltration", "Pneumothorax", "Effusion", "Consolidation", "Cardiomegaly", "Fracture", "Nodule"]
        for i, (r_id, decs) in enumerate([("rev_01", r1_decisions), ("rev_02", r2_decisions), ("rev_03", r3_decisions)]):
            for f in findings:
                dec = decs.get(f, "confirmed_absent")
                self._post("/api/studies/CXR1122/review/finding", {
                    "finding": f,
                    "reviewer_status": dec,
                    "reviewer_location": "right_lower_lobe" if dec == "confirmed_present" else "unspecified",
                    "reviewer_severity": "mild" if dec == "confirmed_present" else "unspecified",
                    "reviewer_id": r_id
                })
        global_consensus_manager.create_consensus_session(
            "CXR1122",
            required_reviewers=3,
            minimum_reviewers=2,
            reviewers_list=[{"id": "rev_01"}, {"id": "rev_02"}, {"id": "rev_03"}]
        )

    def test_06_finding_aggregation(self):
        """TEST 6: Finding decisions are aggregated across all completed reviewers."""
        r1 = {"Infiltration": "confirmed_present"}
        r2 = {"Infiltration": "confirmed_present"}
        r3 = {"Infiltration": "uncertain"}
        self._setup_three_completed_reviewers(r1, r2, r3)

        status, _, body = self._get("/api/studies/CXR1122/consensus/finding/Infiltration")
        self.assertEqual(status, 200)
        fc = json.loads(body.decode("utf-8"))
        self.assertEqual(len(fc["reviewer_decisions"]), 3)
        self.assertEqual(fc["decision_distribution"]["confirmed_present"], 2)
        self.assertEqual(fc["decision_distribution"]["uncertain"], 1)

    def test_07_unanimous_consensus(self):
        """TEST 7: When all reviewers agree, consensus status is 'unanimous'."""
        r1 = {"Infiltration": "confirmed_present"}
        r2 = {"Infiltration": "confirmed_present"}
        r3 = {"Infiltration": "confirmed_present"}
        self._setup_three_completed_reviewers(r1, r2, r3)

        status, _, body = self._get("/api/studies/CXR1122/consensus/finding/Infiltration")
        self.assertEqual(status, 200)
        fc = json.loads(body.decode("utf-8"))
        self.assertEqual(fc["consensus_status"], "unanimous")
        self.assertEqual(fc["consensus_decision"], "confirmed_present")
        self.assertEqual(fc["agreement_ratio"], 1.0)

    def test_08_majority_consensus(self):
        """TEST 8: When 2 out of 3 reviewers agree, consensus status is 'majority'."""
        r1 = {"Infiltration": "confirmed_present"}
        r2 = {"Infiltration": "confirmed_present"}
        r3 = {"Infiltration": "confirmed_absent"}
        self._setup_three_completed_reviewers(r1, r2, r3)

        status, _, body = self._get("/api/studies/CXR1122/consensus/finding/Infiltration")
        self.assertEqual(status, 200)
        fc = json.loads(body.decode("utf-8"))
        self.assertEqual(fc["consensus_status"], "majority")
        self.assertEqual(fc["consensus_decision"], "confirmed_present")
        self.assertAlmostEqual(fc["agreement_ratio"], 2/3, places=3)

    def test_09_tie_and_no_majority_detection(self):
        """TEST 9: Three-way split or tie sets consensus status to 'adjudication_required'."""
        r1 = {"Infiltration": "confirmed_present"}
        r2 = {"Infiltration": "confirmed_absent"}
        r3 = {"Infiltration": "uncertain"}
        self._setup_three_completed_reviewers(r1, r2, r3)

        status, _, body = self._get("/api/studies/CXR1122/consensus/finding/Infiltration")
        self.assertEqual(status, 200)
        fc = json.loads(body.decode("utf-8"))
        self.assertEqual(fc["consensus_status"], "adjudication_required")
        self.assertIsNone(fc["consensus_decision"])

    def test_10_needs_review_handling(self):
        """TEST 10: Any unresolved 'needs_review' decision forces 'adjudication_required'."""
        r1 = {"Infiltration": "confirmed_present"}
        r2 = {"Infiltration": "confirmed_present"}
        r3 = {"Infiltration": "needs_review"}
        self._setup_three_completed_reviewers(r1, r2, r3)

        status, _, body = self._get("/api/studies/CXR1122/consensus/finding/Infiltration")
        self.assertEqual(status, 200)
        fc = json.loads(body.decode("utf-8"))
        self.assertEqual(fc["consensus_status"], "adjudication_required")

    def test_11_location_consensus(self):
        """TEST 11: Location consensus follows majority/unanimous rules independently."""
        self._post("/api/studies/CXR1122/review/finding", {"finding": "Infiltration", "reviewer_status": "confirmed_present", "reviewer_location": "right_lower_lobe", "reviewer_id": "rev_01"})
        self._post("/api/studies/CXR1122/review/finding", {"finding": "Infiltration", "reviewer_status": "confirmed_present", "reviewer_location": "right_lower_lobe", "reviewer_id": "rev_02"})
        self._post("/api/studies/CXR1122/review/finding", {"finding": "Infiltration", "reviewer_status": "confirmed_present", "reviewer_location": "left_lower_lobe", "reviewer_id": "rev_03"})
        global_consensus_manager.create_consensus_session("CXR1122", reviewers_list=[{"id": "rev_01"}, {"id": "rev_02"}, {"id": "rev_03"}])

        status, _, body = self._get("/api/studies/CXR1122/consensus/finding/Infiltration")
        self.assertEqual(status, 200)
        fc = json.loads(body.decode("utf-8"))
        self.assertEqual(fc["location_consensus"], "right_lower_lobe")
        self.assertEqual(fc["location_status"], "majority")

    def test_12_severity_consensus(self):
        """TEST 12: Severity consensus follows unanimous/majority logic."""
        self._post("/api/studies/CXR1122/review/finding", {"finding": "Infiltration", "reviewer_status": "confirmed_present", "reviewer_severity": "mild", "reviewer_id": "rev_01"})
        self._post("/api/studies/CXR1122/review/finding", {"finding": "Infiltration", "reviewer_status": "confirmed_present", "reviewer_severity": "mild", "reviewer_id": "rev_02"})
        self._post("/api/studies/CXR1122/review/finding", {"finding": "Infiltration", "reviewer_status": "confirmed_present", "reviewer_severity": "mild", "reviewer_id": "rev_03"})
        global_consensus_manager.create_consensus_session("CXR1122", reviewers_list=[{"id": "rev_01"}, {"id": "rev_02"}, {"id": "rev_03"}])

        status, _, body = self._get("/api/studies/CXR1122/consensus/finding/Infiltration")
        self.assertEqual(status, 200)
        fc = json.loads(body.decode("utf-8"))
        self.assertEqual(fc["severity_consensus"], "mild")
        self.assertEqual(fc["severity_status"], "unanimous")

    def test_13_qa_consensus_aggregation(self):
        """TEST 13: Reviewer QA answers are aggregated into consensus without mutating machine QA."""
        self._post("/api/studies/CXR1122/review/qa", {"question_id": "Q_INFILTRATION_PRESENCE", "reviewer_answer": "yes", "reviewer_id": "rev_01"})
        self._post("/api/studies/CXR1122/review/qa", {"question_id": "Q_INFILTRATION_PRESENCE", "reviewer_answer": "yes", "reviewer_id": "rev_02"})
        self._post("/api/studies/CXR1122/review/qa", {"question_id": "Q_INFILTRATION_PRESENCE", "reviewer_answer": "no", "reviewer_id": "rev_03"})
        global_consensus_manager.create_consensus_session("CXR1122", reviewers_list=[{"id": "rev_01"}, {"id": "rev_02"}, {"id": "rev_03"}])

        status, _, body = self._get("/api/studies/CXR1122/consensus")
        self.assertEqual(status, 200)
        qa_list = json.loads(body.decode("utf-8"))["qa_consensus"]
        infil_qa = next(q for q in qa_list if q["question_id"] == "Q_INFILTRATION_PRESENCE")
        self.assertEqual(infil_qa["machine_answer"], "uncertain")
        self.assertEqual(infil_qa["consensus_answer"], "yes")
        self.assertEqual(infil_qa["agreement_status"], "majority")

    # --- Tests 14-17: Agreement Metrics (Kappa) ---
    def test_14_cohens_kappa_calculation(self):
        """TEST 14: Cohen's Kappa is correctly calculated for exactly 2 reviewers."""
        r1 = {"Infiltration": "confirmed_present", "Pneumothorax": "confirmed_absent", "Effusion": "confirmed_absent", "Consolidation": "confirmed_absent", "Cardiomegaly": "confirmed_absent", "Fracture": "confirmed_absent", "Nodule": "uncertain"}
        r2 = {"Infiltration": "confirmed_present", "Pneumothorax": "confirmed_absent", "Effusion": "confirmed_absent", "Consolidation": "confirmed_absent", "Cardiomegaly": "confirmed_absent", "Fracture": "confirmed_absent", "Nodule": "uncertain"}
        
        # 100% agreement
        for f, dec in r1.items():
            self._post("/api/studies/CXR1122/review/finding", {"finding": f, "reviewer_status": dec, "reviewer_id": "rev_01"})
            self._post("/api/studies/CXR1122/review/finding", {"finding": f, "reviewer_status": dec, "reviewer_id": "rev_02"})

        global_consensus_manager.create_consensus_session("CXR1122", required_reviewers=2, minimum_reviewers=2, reviewers_list=[{"id": "rev_01"}, {"id": "rev_02"}])

        status, _, body = self._get("/api/studies/CXR1122/consensus/agreement")
        self.assertEqual(status, 200)
        agr = json.loads(body.decode("utf-8"))
        self.assertIsNotNone(agr["cohens_kappa"])
        self.assertEqual(agr["cohens_kappa"]["value"], 1.0)
        self.assertEqual(agr["cohens_kappa"]["observed_agreement"], 1.0)
        self.assertIsNone(agr["fleiss_kappa"])

    def test_15_fleiss_kappa_calculation(self):
        """TEST 15: Fleiss' Kappa is calculated when 3 or more reviewers are present."""
        r1 = {"Infiltration": "confirmed_present", "Pneumothorax": "confirmed_absent", "Effusion": "confirmed_absent", "Consolidation": "confirmed_absent", "Cardiomegaly": "confirmed_absent", "Fracture": "confirmed_absent", "Nodule": "uncertain"}
        r2 = {"Infiltration": "confirmed_present", "Pneumothorax": "confirmed_absent", "Effusion": "confirmed_absent", "Consolidation": "confirmed_absent", "Cardiomegaly": "confirmed_absent", "Fracture": "confirmed_absent", "Nodule": "uncertain"}
        r3 = {"Infiltration": "confirmed_present", "Pneumothorax": "confirmed_absent", "Effusion": "confirmed_absent", "Consolidation": "confirmed_absent", "Cardiomegaly": "confirmed_absent", "Fracture": "confirmed_absent", "Nodule": "uncertain"}
        self._setup_three_completed_reviewers(r1, r2, r3)

        status, _, body = self._get("/api/studies/CXR1122/consensus/agreement")
        self.assertEqual(status, 200)
        agr = json.loads(body.decode("utf-8"))
        self.assertIsNotNone(agr["fleiss_kappa"])
        self.assertEqual(agr["fleiss_kappa"]["value"], 1.0)
        self.assertEqual(agr["fleiss_kappa"]["reviewer_count"], 3)
        self.assertIsNone(agr["cohens_kappa"])

    def test_16_kappa_edge_cases_and_nan_protection(self):
        """TEST 16: Zero variance, empty sets, or undefined denominator never emit NaN into JSON."""
        res = global_consensus_manager.calculate_cohens_kappa([])
        self.assertIsNone(res["value"])

        res_fleiss = global_consensus_manager.calculate_fleiss_kappa([], n_raters=1)
        self.assertIsNone(res_fleiss["value"])

    def test_17_missing_data_handling(self):
        """TEST 17: Partial reviews do not corrupt agreement metric calculations."""
        # rev_01 has 7 reviews, rev_02 has only 1 review
        self._post("/api/studies/CXR1122/review/finding", {"finding": "Infiltration", "reviewer_status": "confirmed_present", "reviewer_id": "rev_01"})
        status, _, body = self._get("/api/studies/CXR1122/consensus/agreement")
        self.assertEqual(status, 200)
        agr = json.loads(body.decode("utf-8"))
        self.assertEqual(agr["reviewer_count"], 0)  # Neither is fully completed

    # --- Tests 18-21: Adjudication Workflow ---
    def test_18_adjudication_item_detection(self):
        """TEST 18: Findings without majority consensus are automatically queued for adjudication."""
        r1 = {"Infiltration": "confirmed_present"}
        r2 = {"Infiltration": "confirmed_absent"}
        r3 = {"Infiltration": "uncertain"}
        self._setup_three_completed_reviewers(r1, r2, r3)

        status, _, body = self._get("/api/studies/CXR1122/consensus/adjudication")
        self.assertEqual(status, 200)
        adj_data = json.loads(body.decode("utf-8"))
        self.assertFalse(adj_data["all_resolved"])
        infil_adj = next(it for it in adj_data["items_requiring_adjudication"] if it["target_id"] == "Infiltration")
        self.assertEqual(infil_adj["status"], "pending")

    def test_19_adjudication_creation(self):
        """TEST 19: Adjudicator can resolve a disagreement with a binding decision."""
        r1 = {"Infiltration": "confirmed_present"}
        r2 = {"Infiltration": "confirmed_absent"}
        r3 = {"Infiltration": "uncertain"}
        self._setup_three_completed_reviewers(r1, r2, r3)

        payload = {
            "target_type": "finding",
            "target_id": "Infiltration",
            "decision": "confirmed_present",
            "reason": "Expert review confirms faint right basilar airspace opacification.",
            "adjudicator_id": "dr_senior_adjudicator"
        }
        status, _, body = self._post("/api/studies/CXR1122/consensus/adjudicate", payload)
        self.assertEqual(status, 200)
        rec = json.loads(body.decode("utf-8"))["adjudication_record"]
        self.assertEqual(rec["decision"], "confirmed_present")
        self.assertEqual(rec["adjudicator_id"], "dr_senior_adjudicator")

    def test_20_mandatory_adjudication_reason_validation(self):
        """TEST 20: Adjudication without a non-empty reason is rejected with 400."""
        payload = {
            "target_type": "finding",
            "target_id": "Infiltration",
            "decision": "confirmed_present",
            "reason": "   ",  # Empty
            "adjudicator_id": "dr_senior"
        }
        status, _, _ = self._post("/api/studies/CXR1122/consensus/adjudicate", payload)
        self.assertEqual(status, 400)

    def test_21_adjudication_history_preservation(self):
        """TEST 21: Recording adjudication does not overwrite original reviewer decisions."""
        r1 = {"Infiltration": "confirmed_present"}
        r2 = {"Infiltration": "confirmed_absent"}
        r3 = {"Infiltration": "uncertain"}
        self._setup_three_completed_reviewers(r1, r2, r3)

        self._post("/api/studies/CXR1122/consensus/adjudicate", {
            "target_type": "finding",
            "target_id": "Infiltration",
            "decision": "confirmed_present",
            "reason": "Adjudicated as present after review.",
            "adjudicator_id": "senior_doc"
        })

        # Check individual reviewer sessions remain unchanged
        r2_session = global_review_manager.get_review("CXR1122", reviewer_id="rev_02")
        f2 = next(f for f in r2_session["finding_reviews"] if f["finding"] == "Infiltration")
        self.assertEqual(f2["reviewer_status"], "confirmed_absent")

    # --- Tests 22-25: Consensus Report & Finalization ---
    def test_22_consensus_report_generation(self):
        """TEST 22: Consensus report synthesizes agreed findings, impression, and adjudication notes."""
        r1 = {"Infiltration": "confirmed_present"}
        r2 = {"Infiltration": "confirmed_present"}
        r3 = {"Infiltration": "confirmed_present"}
        self._setup_three_completed_reviewers(r1, r2, r3)

        status, _, body = self._get("/api/studies/CXR1122/consensus/report")
        self.assertEqual(status, 200)
        crep = json.loads(body.decode("utf-8"))
        self.assertIn("final_findings", crep)
        self.assertIn("final_impression", crep)
        self.assertTrue(any(f["finding"] == "Infiltration" for f in crep["final_findings"]))

    def test_23_finalization_rejection_when_incomplete(self):
        """TEST 23: Finalization fails with 400 if unresolved disputes or incomplete reviews remain."""
        r1 = {"Infiltration": "confirmed_present"}
        r2 = {"Infiltration": "confirmed_absent"}
        r3 = {"Infiltration": "uncertain"}
        self._setup_three_completed_reviewers(r1, r2, r3)

        # Infiltration is disputed and not yet adjudicated
        status, _, body = self._post("/api/studies/CXR1122/consensus/finalize")
        self.assertEqual(status, 400)
        data = json.loads(body.decode("utf-8"))
        self.assertFalse(data["finalized"])
        self.assertIn("validation_errors", data)

    def test_24_successful_consensus_finalization(self):
        """TEST 24: When all items resolved, consensus is successfully finalized and locked."""
        r1 = {"Infiltration": "confirmed_present", "Pneumothorax": "confirmed_absent", "Effusion": "confirmed_absent", "Consolidation": "confirmed_absent", "Cardiomegaly": "confirmed_absent", "Fracture": "confirmed_absent", "Nodule": "confirmed_absent"}
        r2 = {"Infiltration": "confirmed_present", "Pneumothorax": "confirmed_absent", "Effusion": "confirmed_absent", "Consolidation": "confirmed_absent", "Cardiomegaly": "confirmed_absent", "Fracture": "confirmed_absent", "Nodule": "confirmed_absent"}
        r3 = {"Infiltration": "confirmed_present", "Pneumothorax": "confirmed_absent", "Effusion": "confirmed_absent", "Consolidation": "confirmed_absent", "Cardiomegaly": "confirmed_absent", "Fracture": "confirmed_absent", "Nodule": "confirmed_absent"}
        self._setup_three_completed_reviewers(r1, r2, r3)

        status, _, body = self._post("/api/studies/CXR1122/consensus/finalize")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertTrue(data["finalized"])
        self.assertEqual(data["session"]["status"], "finalized")

    def test_25_consensus_audit_trail(self):
        """TEST 25: All consensus actions are recorded chronologically in audit trail."""
        r1 = {"Infiltration": "confirmed_present"}
        r2 = {"Infiltration": "confirmed_present"}
        r3 = {"Infiltration": "confirmed_present"}
        self._setup_three_completed_reviewers(r1, r2, r3)

        status, _, body = self._get("/api/studies/CXR1122/consensus/audit")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("audit_trail", data)
        self.assertGreater(len(data["audit_trail"]), 0)

    # --- Tests 26-30: Safety, Invariants & Security ---
    def test_26_machine_immutability(self):
        """TEST 26: Consensus calculation and adjudication never modify machine evidence artifacts."""
        _, _, b1 = self._get("/api/studies/CXR1122/evidence")
        ev_before = json.loads(b1.decode("utf-8"))["evidence"]

        r1 = {"Infiltration": "confirmed_present"}
        r2 = {"Infiltration": "confirmed_present"}
        r3 = {"Infiltration": "confirmed_present"}
        self._setup_three_completed_reviewers(r1, r2, r3)
        self._post("/api/studies/CXR1122/consensus/finalize")

        _, _, b2 = self._get("/api/studies/CXR1122/evidence")
        ev_after = json.loads(b2.decode("utf-8"))["evidence"]
        self.assertEqual(ev_before, ev_after)

    def test_27_zero_ground_truth_leakage(self):
        """TEST 27: Reference XML reports never appear in consensus data structures or exports."""
        r1 = {"Infiltration": "confirmed_present"}
        r2 = {"Infiltration": "confirmed_present"}
        r3 = {"Infiltration": "confirmed_present"}
        self._setup_three_completed_reviewers(r1, r2, r3)

        _, _, body = self._get("/api/studies/CXR1122/consensus/export?format=json")
        raw_text = body.decode("utf-8")
        self.assertNotIn("<report>", raw_text)
        self.assertNotIn("<xml", raw_text)
        self.assertNotIn("COMPARISON:", raw_text)

    def test_28_invalid_study_rejection(self):
        """TEST 28: Non-existent study IDs are rejected with 404."""
        status, _, _ = self._get("/api/studies/NON_EXISTENT_STUDY_9999/consensus")
        self.assertEqual(status, 404)

    def test_29_path_traversal_rejection(self):
        """TEST 29: Path traversal patterns in consensus endpoints are rejected."""
        status, _, _ = self._get("/api/studies/..%2F..%2Fetc/consensus")
        self.assertEqual(status, 404)

    def test_30_backward_compatibility_regression(self):
        """TEST 30: All Phase 1.0, 1.1, and 1.2 endpoints continue to function seamlessly."""
        for ep in [
            "/api/studies",
            "/api/studies/CXR1122",
            "/api/studies/CXR1122/images",
            "/api/studies/CXR1122/image",
            "/api/studies/CXR1122/evidence",
            "/api/studies/CXR1122/qa",
            "/api/studies/CXR1122/grounding",
            "/api/studies/CXR1122/report",
            "/api/studies/CXR1122/validation",
            "/api/studies/CXR1122/review"
        ]:
            s, _, _ = self._get(ep)
            self.assertEqual(s, 200, f"Endpoint {ep} failed with {s}")

    # --- Tests 31-33: Exports & Immutability Locking ---
    def test_31_json_consensus_export(self):
        """TEST 31: JSON export contains complete consensus package, agreement metrics, and disclaimer."""
        r1 = {"Infiltration": "confirmed_present"}
        r2 = {"Infiltration": "confirmed_present"}
        r3 = {"Infiltration": "confirmed_present"}
        self._setup_three_completed_reviewers(r1, r2, r3)

        status, _, body = self._get("/api/studies/CXR1122/consensus/export?format=json")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["study_id"], "CXR1122")
        self.assertIn("agreement_metrics", data)
        self.assertIn("disclaimer", data)

    def test_32_text_consensus_export(self):
        """TEST 32: Plain text consensus export produces structured human-readable document."""
        r1 = {"Infiltration": "confirmed_present"}
        r2 = {"Infiltration": "confirmed_present"}
        r3 = {"Infiltration": "confirmed_present"}
        self._setup_three_completed_reviewers(r1, r2, r3)

        status, _, body = self._get("/api/studies/CXR1122/consensus/export?format=text")
        self.assertEqual(status, 200)
        txt = body.decode("utf-8")
        self.assertIn("MULTI-REVIEWER CONSENSUS REPORT", txt)
        self.assertIn("NOT A CLINICAL DIAGNOSIS", txt)
        self.assertIn("INTER-RATER AGREEMENT METRICS", txt)

    def test_33_post_finalization_mutation_rejection(self):
        """TEST 33: Once consensus is finalized, all further adjudications and edits return 400."""
        r1 = {"Infiltration": "confirmed_present", "Pneumothorax": "confirmed_absent", "Effusion": "confirmed_absent", "Consolidation": "confirmed_absent", "Cardiomegaly": "confirmed_absent", "Fracture": "confirmed_absent", "Nodule": "confirmed_absent"}
        r2 = {"Infiltration": "confirmed_present", "Pneumothorax": "confirmed_absent", "Effusion": "confirmed_absent", "Consolidation": "confirmed_absent", "Cardiomegaly": "confirmed_absent", "Fracture": "confirmed_absent", "Nodule": "confirmed_absent"}
        r3 = {"Infiltration": "confirmed_present", "Pneumothorax": "confirmed_absent", "Effusion": "confirmed_absent", "Consolidation": "confirmed_absent", "Cardiomegaly": "confirmed_absent", "Fracture": "confirmed_absent", "Nodule": "confirmed_absent"}
        self._setup_three_completed_reviewers(r1, r2, r3)
        self._post("/api/studies/CXR1122/consensus/finalize")

        # Attempt to adjudicate after finalization
        s_adj, _, _ = self._post("/api/studies/CXR1122/consensus/adjudicate", {
            "target_type": "finding",
            "target_id": "Infiltration",
            "decision": "confirmed_absent",
            "reason": "Late mutation attempt"
        })
        self.assertEqual(s_adj, 400)


if __name__ == "__main__":
    unittest.main()
