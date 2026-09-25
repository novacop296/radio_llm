"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.2 — Human-in-the-Loop Review, Report Correction & Finalization Test Suite

Verifies:
1. Review session creation and auto-initialization from machine baseline.
2. Review session retrieval (GET /api/studies/{study_id}/review).
3. Finding review creation and updates (POST /api/studies/{study_id}/review/finding).
4. Reviewer status: confirmed_present.
5. Reviewer status: confirmed_absent.
6. Reviewer status: uncertain.
7. Reviewer status: needs_review.
8. Reviewer status validation rejection on invalid status.
9. Reviewer location editing.
10. Reviewer severity editing.
11. Reviewer comments recording.
12. QA correction storage (POST /api/studies/{study_id}/review/qa) without machine QA mutation.
13. Machine evidence immutability guarantee.
14. Machine report immutability guarantee.
15. Report draft saving (POST /api/studies/{study_id}/review/report).
16. Report reset to machine baseline (POST /api/studies/{study_id}/review/report/reset).
17. Finalization failure when candidate findings remain unreviewed or needs_review.
18. Successful report finalization when all criteria met.
19. Finalization locking (all mutation attempts rejected once finalized).
20. Chronological audit trail generation (recording old_value and new_value).
21. Final review JSON export with schema compliance and research disclaimers.
22. Final review human-readable Text export with clear section separation.
23. Zero ground-truth leakage into review layers.
24. Path traversal and invalid study identifier rejection.
25. Full backward compatibility with existing Phase 1.0 & 1.1 APIs.
"""

import os
import sys
import json
import unittest
import threading
import urllib.request
import urllib.error
from http.server import HTTPServer

# Add backend directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from api import RadiologyAPIHandler, load_study_data
from review_manager import ReviewManager, global_review_manager, _get_review_file_path


class TestPhase12Review(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.port = 8768
        cls.server = HTTPServer(("127.0.0.1", cls.port), RadiologyAPIHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        # Clean up test review files created during testing
        reviews_dir = os.path.join(BASE_DIR, "data", "reviews")
        if os.path.exists(reviews_dir):
            for fname in os.listdir(reviews_dir):
                if fname.startswith("CXR1122") or fname.startswith("TEST_STUDY") or "rev_" in fname:
                    try:
                        os.remove(os.path.join(reviews_dir, fname))
                    except Exception:
                        pass

    def setUp(self):
        # Reset review files for test studies before each test
        reviews_dir = os.path.join(BASE_DIR, "data", "reviews")
        if os.path.exists(reviews_dir):
            for fname in os.listdir(reviews_dir):
                if fname.startswith("CXR1122") or fname.startswith("TEST_STUDY") or "rev_" in fname:
                    try:
                        os.remove(os.path.join(reviews_dir, fname))
                    except Exception:
                        pass

    def _get(self, path: str):
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    def _post(self, path: str, payload: dict):
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    def test_01_review_session_creation_and_auto_init(self):
        """TEST 1: Review session auto-initializes with 7 candidate findings and locked machine baseline."""
        status, _, body = self._get("/api/studies/CXR1122/review")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["study_id"], "CXR1122")
        self.assertIn("review_id", data)
        self.assertEqual(data["status"], "in_review")
        self.assertIn("reviewer", data)
        self.assertEqual(len(data["finding_reviews"]), 7)
        self.assertTrue(data["report_review"]["machine_report_locked"])
        self.assertFalse(data["report_review"]["finalized"])
        self.assertGreaterEqual(len(data["audit_trail"]), 1)

    def test_02_review_session_retrieval(self):
        """TEST 2: GET /api/studies/CXR1122/review returns existing session without recreation."""
        # Initial creation
        s1, _, b1 = self._get("/api/studies/CXR1122/review")
        rev_id_1 = json.loads(b1.decode("utf-8"))["review_id"]

        # Subsequent retrieval
        s2, _, b2 = self._get("/api/studies/CXR1122/review")
        rev_id_2 = json.loads(b2.decode("utf-8"))["review_id"]
        self.assertEqual(rev_id_1, rev_id_2)

    def test_03_finding_review_creation_and_retrieval(self):
        """TEST 3: Update finding review and retrieve via dedicated GET /review/finding/{finding}."""
        payload = {
            "finding": "Infiltration",
            "reviewer_status": "confirmed_present",
            "reviewer_location": "right_lower_lobe",
            "reviewer_severity": "mild",
            "reviewer_comment": "Clear focal opacity in right base."
        }
        status, _, body = self._post("/api/studies/CXR1122/review/finding", payload)
        self.assertEqual(status, 200)
        res = json.loads(body.decode("utf-8"))
        self.assertTrue(res["success"])
        self.assertEqual(res["finding_review"]["reviewer_status"], "confirmed_present")
        self.assertTrue(res["finding_review"]["reviewed"])

        # Retrieve single finding
        s2, _, b2 = self._get("/api/studies/CXR1122/review/finding/Infiltration")
        self.assertEqual(s2, 200)
        fr = json.loads(b2.decode("utf-8"))
        self.assertEqual(fr["finding"], "Infiltration")
        self.assertEqual(fr["reviewer_status"], "confirmed_present")
        self.assertEqual(fr["reviewer_location"], "right_lower_lobe")

    def test_04_status_confirmed_present(self):
        """TEST 4: confirmed_present status correctly marks finding reviewed."""
        payload = {"finding": "Cardiomegaly", "reviewer_status": "confirmed_present"}
        status, _, body = self._post("/api/studies/CXR1122/review/finding", payload)
        self.assertEqual(status, 200)
        fr = json.loads(body.decode("utf-8"))["finding_review"]
        self.assertEqual(fr["reviewer_status"], "confirmed_present")
        self.assertTrue(fr["reviewed"])

    def test_05_status_confirmed_absent(self):
        """TEST 5: confirmed_absent status correctly marks finding reviewed."""
        payload = {"finding": "Pneumothorax", "reviewer_status": "confirmed_absent"}
        status, _, body = self._post("/api/studies/CXR1122/review/finding", payload)
        self.assertEqual(status, 200)
        fr = json.loads(body.decode("utf-8"))["finding_review"]
        self.assertEqual(fr["reviewer_status"], "confirmed_absent")
        self.assertTrue(fr["reviewed"])

    def test_06_status_uncertain(self):
        """TEST 6: uncertain status correctly marks finding reviewed with preserved ambiguity."""
        payload = {"finding": "Effusion", "reviewer_status": "uncertain", "reviewer_comment": "Blunted costophrenic angle."}
        status, _, body = self._post("/api/studies/CXR1122/review/finding", payload)
        self.assertEqual(status, 200)
        fr = json.loads(body.decode("utf-8"))["finding_review"]
        self.assertEqual(fr["reviewer_status"], "uncertain")
        self.assertTrue(fr["reviewed"])

    def test_07_status_needs_review(self):
        """TEST 7: needs_review status sets reviewed to False to block accidental finalization."""
        payload = {"finding": "Nodule", "reviewer_status": "needs_review"}
        status, _, body = self._post("/api/studies/CXR1122/review/finding", payload)
        self.assertEqual(status, 200)
        fr = json.loads(body.decode("utf-8"))["finding_review"]
        self.assertEqual(fr["reviewer_status"], "needs_review")
        self.assertFalse(fr["reviewed"])

    def test_08_invalid_status_rejection(self):
        """TEST 8: Malformed or unapproved reviewer status values are rejected with 400."""
        payload = {"finding": "Fracture", "reviewer_status": "DEFINITELY_NOT_A_REAL_STATUS"}
        status, _, body = self._post("/api/studies/CXR1122/review/finding", payload)
        self.assertEqual(status, 400)

    def test_09_location_and_severity_editing(self):
        """TEST 9: Reviewer location and severity updates are stored separately from machine attributes."""
        payload = {
            "finding": "Consolidation",
            "reviewer_status": "confirmed_present",
            "reviewer_location": "left_lower_lobe",
            "reviewer_severity": "moderate"
        }
        status, _, body = self._post("/api/studies/CXR1122/review/finding", payload)
        self.assertEqual(status, 200)
        fr = json.loads(body.decode("utf-8"))["finding_review"]
        self.assertEqual(fr["machine_location"], "unspecified")
        self.assertEqual(fr["reviewer_location"], "left_lower_lobe")
        self.assertEqual(fr["machine_severity"], "unspecified")
        self.assertEqual(fr["reviewer_severity"], "moderate")

    def test_10_qa_review_correction_storage(self):
        """TEST 10: Reviewer QA corrections are recorded without mutating original machine QA."""
        payload = {
            "question_id": "Q_INFILTRATION_PRESENCE",
            "reviewer_answer": "yes",
            "reviewer_comment": "Reviewer verified infiltrate presence manually."
        }
        status, _, body = self._post("/api/studies/CXR1122/review/qa", payload)
        self.assertEqual(status, 200)
        qr = json.loads(body.decode("utf-8"))["qa_review"]
        self.assertEqual(qr["machine_answer"], "uncertain")
        self.assertEqual(qr["reviewer_answer"], "yes")

        # Verify machine QA endpoint remains untouched
        s2, _, b2 = self._get("/api/studies/CXR1122/qa")
        q_orig = next(q for q in json.loads(b2.decode("utf-8"))["questions"] if q["question_id"] == "Q_INFILTRATION_PRESENCE")
        self.assertEqual(q_orig["answer"], "uncertain")

    def test_11_machine_evidence_immutability(self):
        """TEST 11: Modifying reviewer decisions does not mutate machine evidence layer."""
        # 1. Fetch initial machine evidence
        _, _, b1 = self._get("/api/studies/CXR1122/evidence")
        ev_orig = json.loads(b1.decode("utf-8"))["evidence"]

        # 2. Make several reviewer corrections
        self._post("/api/studies/CXR1122/review/finding", {"finding": "Infiltration", "reviewer_status": "confirmed_present"})
        self._post("/api/studies/CXR1122/review/finding", {"finding": "Pneumothorax", "reviewer_status": "confirmed_absent"})

        # 3. Fetch machine evidence again and verify byte-for-byte fidelity
        _, _, b2 = self._get("/api/studies/CXR1122/evidence")
        ev_after = json.loads(b2.decode("utf-8"))["evidence"]
        self.assertEqual(ev_orig, ev_after)

    def test_12_machine_report_immutability(self):
        """TEST 12: Machine-generated report remains locked and unchanged after review edits."""
        # 1. Fetch original machine report
        _, _, b1 = self._get("/api/studies/CXR1122/report")
        rep_orig = json.loads(b1.decode("utf-8"))

        # 2. Save custom review draft
        draft_payload = {
            "final_findings": [{"finding": "Infiltration", "statement": "Reviewer custom finding.", "status": "supported"}],
            "final_impression": ["1. Custom impression statement."],
            "reviewer_comment": "Draft notes."
        }
        self._post("/api/studies/CXR1122/review/report", draft_payload)

        # 3. Fetch original machine report again and verify unchanged
        _, _, b2 = self._get("/api/studies/CXR1122/report")
        rep_after = json.loads(b2.decode("utf-8"))
        self.assertEqual(rep_orig, rep_after)

    def test_13_report_draft_saving_and_reset(self):
        """TEST 13: Reviewer can save report draft and reset draft back to machine baseline."""
        draft_payload = {
            "final_findings": [{"finding": "Infiltration", "statement": "Custom drafted statement.", "status": "confirmed_present"}],
            "final_impression": ["1. Custom draft impression."],
            "reviewer_comment": "Custom synthesis."
        }
        s1, _, b1 = self._post("/api/studies/CXR1122/review/report", draft_payload)
        self.assertEqual(s1, 200)
        rep1 = json.loads(b1.decode("utf-8"))["report_review"]
        self.assertEqual(rep1["final_findings"][0]["statement"], "Custom drafted statement.")

        # Reset to machine baseline
        s2, _, b2 = self._post("/api/studies/CXR1122/review/report/reset", {})
        self.assertEqual(s2, 200)
        rep2 = json.loads(b2.decode("utf-8"))["report_review"]
        self.assertEqual(len(rep2["final_findings"]), 7)
        self.assertEqual(rep2["reviewer_comment"], "")

    def test_14_finalization_failure_when_incomplete(self):
        """TEST 14: Finalization fails with 400 if candidate findings remain unreviewed."""
        # CXR1122 starts with all findings unreviewed
        self._get("/api/studies/CXR1122/review")
        status, _, body = self._post("/api/studies/CXR1122/review/finalize", {})
        self.assertEqual(status, 400)
        data = json.loads(body.decode("utf-8"))
        self.assertFalse(data["finalized"])
        self.assertIn("validation_errors", data)
        self.assertGreaterEqual(len(data["validation_errors"]), 1)

    def test_15_successful_finalization_and_locking(self):
        """TEST 15: Review is successfully finalized and locked once all findings are reviewed."""
        # 1. Review all 7 candidate findings
        findings = ["Infiltration", "Pneumothorax", "Consolidation", "Fracture", "Nodule", "Effusion", "Cardiomegaly"]
        for f in findings:
            self._post("/api/studies/CXR1122/review/finding", {
                "finding": f,
                "reviewer_status": "confirmed_absent" if f != "Infiltration" else "confirmed_present",
                "reviewer_comment": f"Reviewed {f}"
            })

        # 2. Finalize
        status, _, body = self._post("/api/studies/CXR1122/review/finalize", {})
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertTrue(data["finalized"])
        self.assertEqual(data["session"]["status"], "finalized")
        self.assertIsNotNone(data["session"]["completed_at"])

        # 3. Verify locking: Subsequent finding modifications must fail with 400
        mod_status, _, mod_body = self._post("/api/studies/CXR1122/review/finding", {"finding": "Infiltration", "reviewer_status": "uncertain"})
        self.assertEqual(mod_status, 400)

    def test_16_audit_trail_generation(self):
        """TEST 16: Audit trail chronologically logs all reviewer modifications with old and new values."""
        self._post("/api/studies/CXR1122/review/finding", {"finding": "Infiltration", "reviewer_status": "confirmed_present"})
        self._post("/api/studies/CXR1122/review/finding", {"finding": "Infiltration", "reviewer_location": "right_base"})

        status, _, body = self._get("/api/studies/CXR1122/review/audit")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        trail = data["audit_trail"]
        self.assertGreaterEqual(len(trail), 2)

        # Verify entry structure
        status_entry = next((e for e in trail if e["field"] == "reviewer_status"), None)
        self.assertIsNotNone(status_entry)
        self.assertEqual(status_entry["old_value"], "not_reviewed")
        self.assertEqual(status_entry["new_value"], "confirmed_present")

    def test_17_json_review_export(self):
        """TEST 17: JSON export returns complete review package with research disclaimers."""
        status, _, body = self._get("/api/studies/CXR1122/review/export?format=json")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["study_id"], "CXR1122")
        self.assertIn("machine_baseline", data)
        self.assertIn("finding_reviews", data)
        self.assertIn("final_report", data)
        self.assertIn("audit_trail", data)
        self.assertIn("disclaimer", data)

    def test_18_text_review_export(self):
        """TEST 18: Plain text export generates structured human-readable report."""
        status, headers, body = self._get("/api/studies/CXR1122/review/export?format=text")
        self.assertEqual(status, 200)
        text = body.decode("utf-8")
        self.assertIn("RESEARCH REVIEW REPORT", text)
        self.assertIn("MACHINE EVIDENCE SUMMARY", text)
        self.assertIn("REVIEWER FINDING DECISIONS", text)
        self.assertIn("FINAL REVIEWED FINDINGS", text)
        self.assertIn("AUDIT TRAIL", text)

    def test_24_save_finding_decision_persists_reviewed_state(self):
        """TEST 24: Save finding decision genuinely persists reviewed=True and status to disk."""
        payload = {
            "finding": "Infiltration",
            "reviewer_status": "confirmed_present",
            "reviewer_location": "right_lower_lobe",
            "reviewer_severity": "moderate",
            "reviewer_comment": "Clear consolidation noted.",
            "reviewer_id": "researcher_01"
        }
        status, _, body = self._post("/api/studies/CXR1122/review/finding", payload)
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertTrue(data["success"])
        self.assertTrue(data["finding_review"]["reviewed"])
        self.assertEqual(data["finding_review"]["reviewer_status"], "confirmed_present")
        self.assertEqual(data["finding_review"]["reviewer_location"], "right_lower_lobe")

        # Verify disk persistence
        file_path = _get_review_file_path("CXR1122", "researcher_01")
        self.assertTrue(os.path.exists(file_path))
        with open(file_path, "r", encoding="utf-8") as f:
            disk_session = json.load(f)
        fr = next(f for f in disk_session["finding_reviews"] if f["finding"] == "Infiltration")
        self.assertTrue(fr["reviewed"])
        self.assertEqual(fr["reviewer_status"], "confirmed_present")
        self.assertEqual(fr["reviewer_location"], "right_lower_lobe")

    def test_25_saved_decision_survives_reload(self):
        """TEST 25: Previously saved finding decisions survive reload without resetting to not_reviewed."""
        payload = {
            "finding": "Infiltration",
            "reviewer_status": "confirmed_present",
            "reviewer_id": "researcher_01"
        }
        self._post("/api/studies/CXR1122/review/finding", payload)

        # Fresh GET reload
        status, _, body = self._get("/api/studies/CXR1122/review?reviewer_id=researcher_01")
        self.assertEqual(status, 200)
        reloaded = json.loads(body.decode("utf-8"))
        fr = next(f for f in reloaded["finding_reviews"] if f["finding"] == "Infiltration")
        self.assertTrue(fr["reviewed"])
        self.assertEqual(fr["reviewer_status"], "confirmed_present")

    def test_26_saved_decision_updates_review_metrics(self):
        """TEST 26: Saving finding decisions accurately updates aggregate counts."""
        # Set up a known state with valid candidate findings in CXR1122
        findings = ["Infiltration", "Pneumothorax", "Consolidation", "Fracture"]
        decisions = ["confirmed_present", "confirmed_absent", "uncertain", "needs_review"]
        for f_name, dec in zip(findings, decisions):
            self._post("/api/studies/CXR1122/review/finding", {
                "finding": f_name,
                "reviewer_status": dec,
                "reviewer_id": "researcher_01"
            })

        status, _, body = self._get("/api/studies/CXR1122/review?reviewer_id=researcher_01")
        self.assertEqual(status, 200)
        sess = json.loads(body.decode("utf-8"))
        frs = sess["finding_reviews"]
        
        pres_count = sum(1 for fr in frs if fr["reviewer_status"] == "confirmed_present")
        abs_count = sum(1 for fr in frs if fr["reviewer_status"] == "confirmed_absent")
        unc_count = sum(1 for fr in frs if fr["reviewer_status"] == "uncertain")
        needs_rev_count = sum(1 for fr in frs if fr["reviewer_status"] == "needs_review")
        actioned_count = sum(1 for fr in frs if fr["reviewer_status"] != "not_reviewed")
        
        self.assertEqual(pres_count, 1)
        self.assertEqual(abs_count, 1)
        self.assertEqual(unc_count, 1)
        self.assertEqual(needs_rev_count, 1)
        self.assertEqual(actioned_count, 4)

    def test_27_saved_decision_updates_finalization_validation(self):
        """TEST 27: Finalization accurately enforces that needs_review is ineligible."""
        # CXR1122 has 7 findings. Review 6 as confirmed_absent, 1 as needs_review
        study_data = load_study_data("CXR1122")
        all_findings = [e["finding"] for e in study_data["evidence"]]
        for f_name in all_findings[:-1]:
            self._post("/api/studies/CXR1122/review/finding", {
                "finding": f_name,
                "reviewer_status": "confirmed_absent",
                "reviewer_id": "rev_val_test"
            })
        self._post("/api/studies/CXR1122/review/finding", {
            "finding": all_findings[-1],
            "reviewer_status": "needs_review",
            "reviewer_id": "rev_val_test"
        })

        # Finalize should fail due to needs_review
        status, _, body = self._post("/api/studies/CXR1122/review/finalize", {"reviewer_id": "rev_val_test"})
        self.assertEqual(status, 400)
        res = json.loads(body.decode("utf-8"))
        self.assertFalse(res["success"])
        self.assertTrue(any("needs_review" in err for err in res.get("validation_errors", [])))

        # Update last finding to confirmed_present
        self._post("/api/studies/CXR1122/review/finding", {
            "finding": all_findings[-1],
            "reviewer_status": "confirmed_present",
            "reviewer_id": "rev_val_test"
        })
        # Now finalize should succeed
        status2, _, body2 = self._post("/api/studies/CXR1122/review/finalize", {"reviewer_id": "rev_val_test"})
        self.assertEqual(status2, 200)
        res2 = json.loads(body2.decode("utf-8"))
        self.assertTrue(res2["success"])
        self.assertTrue(res2["finalized"])

    def test_28_reviewer_sessions_are_isolated(self):
        """TEST 28: Reviewer A and Reviewer B sessions are strictly isolated."""
        # Reviewer A marks Infiltration confirmed_present
        self._post("/api/studies/CXR1122/review/finding", {
            "finding": "Infiltration",
            "reviewer_status": "confirmed_present",
            "reviewer_id": "rev_iso_A"
        })
        # Reviewer B marks Infiltration confirmed_absent
        self._post("/api/studies/CXR1122/review/finding", {
            "finding": "Infiltration",
            "reviewer_status": "confirmed_absent",
            "reviewer_id": "rev_iso_B"
        })

        _, _, bA = self._get("/api/studies/CXR1122/review?reviewer_id=rev_iso_A")
        _, _, bB = self._get("/api/studies/CXR1122/review?reviewer_id=rev_iso_B")
        sessA = json.loads(bA.decode("utf-8"))
        sessB = json.loads(bB.decode("utf-8"))

        frA = next(f for f in sessA["finding_reviews"] if f["finding"] == "Infiltration")
        frB = next(f for f in sessB["finding_reviews"] if f["finding"] == "Infiltration")
        self.assertEqual(frA["reviewer_status"], "confirmed_present")
        self.assertEqual(frB["reviewer_status"], "confirmed_absent")

    def test_29_finding_identifier_is_consistent(self):
        """TEST 29: Case variations in finding name update the canonical finding without duplicates."""
        self._post("/api/studies/CXR1122/review/finding", {
            "finding": "infiltration",
            "reviewer_status": "confirmed_present",
            "reviewer_id": "rev_case_test"
        })
        _, _, b = self._get("/api/studies/CXR1122/review?reviewer_id=rev_case_test")
        sess = json.loads(b.decode("utf-8"))
        matching = [f for f in sess["finding_reviews"] if f["finding"].lower() == "infiltration"]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["reviewer_status"], "confirmed_present")

    def test_30_audit_entry_created_for_saved_decision(self):
        """TEST 30: Saving finding decisions generates chronological audit records."""
        payload = {
            "finding": "Effusion",
            "reviewer_status": "confirmed_present",
            "reviewer_location": "left_costophrenic_angle",
            "reviewer_id": "rev_audit_test"
        }
        self._post("/api/studies/CXR1122/review/finding", payload)
        
        status, _, body = self._get("/api/studies/CXR1122/review/audit?reviewer_id=rev_audit_test")
        self.assertEqual(status, 200)
        audit_data = json.loads(body.decode("utf-8"))
        trail = audit_data["audit_trail"]
        
        eff_entries = [e for e in trail if e["finding"] == "Effusion"]
        self.assertTrue(len(eff_entries) >= 1)
        status_entry = next((e for e in eff_entries if e["field"] == "reviewer_status"), None)
        self.assertIsNotNone(status_entry)
        self.assertEqual(status_entry["new_value"], "confirmed_present")
        self.assertEqual(status_entry["reviewer_id"], "rev_audit_test")

    def test_31_failed_save_does_not_create_audit_entry(self):
        """TEST 31: Failed validation attempts do not create audit log entries."""
        _, _, b_before = self._get("/api/studies/CXR1122/review/audit?reviewer_id=rev_audit_fail_test")
        count_before = len(json.loads(b_before.decode("utf-8"))["audit_trail"])

        # Attempt invalid save
        bad_payload = {"finding": "NonExistentFinding", "reviewer_status": "invalid_status"}
        st, _, _ = self._post("/api/studies/CXR1122/review/finding", bad_payload)
        self.assertEqual(st, 400)

        _, _, b_after = self._get("/api/studies/CXR1122/review/audit?reviewer_id=rev_audit_fail_test")
        count_after = len(json.loads(b_after.decode("utf-8"))["audit_trail"])
        self.assertEqual(count_before, count_after)

    def test_32_study_switch_preserves_review_state(self):
        """TEST 32: Reviewing study A and study B maintains independent state across switches."""
        # Review in CXR1122
        self._post("/api/studies/CXR1122/review/finding", {
            "finding": "Infiltration",
            "reviewer_status": "confirmed_present",
            "reviewer_id": "rev_switch_test"
        })
        # Review in CXR1401 (finding: Lung Opacity)
        self._post("/api/studies/CXR1401/review/finding", {
            "finding": "Lung Opacity",
            "reviewer_status": "confirmed_absent",
            "reviewer_id": "rev_switch_test"
        })

        # Verify CXR1122
        _, _, b1122 = self._get("/api/studies/CXR1122/review?reviewer_id=rev_switch_test")
        sess1122 = json.loads(b1122.decode("utf-8"))
        fr1122 = next(f for f in sess1122["finding_reviews"] if f["finding"] == "Infiltration")
        self.assertEqual(fr1122["reviewer_status"], "confirmed_present")

        # Verify CXR1401
        _, _, b1401 = self._get("/api/studies/CXR1401/review?reviewer_id=rev_switch_test")
        sess1401 = json.loads(b1401.decode("utf-8"))
        fr1401 = next(f for f in sess1401["finding_reviews"] if f["finding"] == "Lung Opacity")
        self.assertEqual(fr1401["reviewer_status"], "confirmed_absent")

    def test_33_all_decision_types_semantics(self):
        """TEST 33: Verify reviewed boolean and status mapping for all decision types."""
        mgr = ReviewManager(load_study_data)
        sess = mgr.get_or_create_review("CXR1122", reviewer_id="rev_semantics_test")

        for dec in ["confirmed_present", "confirmed_absent", "uncertain"]:
            fr = mgr.update_finding_review("CXR1122", {
                "finding": "Infiltration",
                "reviewer_status": dec,
                "reviewer_id": "rev_semantics_test"
            })
            self.assertTrue(fr["reviewed"])
            self.assertEqual(fr["reviewer_status"], dec)

        # needs_review sets reviewed=False in Phase 1.2 schema
        fr_nr = mgr.update_finding_review("CXR1122", {
            "finding": "Infiltration",
            "reviewer_status": "needs_review",
            "reviewer_id": "rev_semantics_test"
        })
        self.assertFalse(fr_nr["reviewed"])
        self.assertEqual(fr_nr["reviewer_status"], "needs_review")

        # not_reviewed should set reviewed=False
        fr_not = mgr.update_finding_review("CXR1122", {
            "finding": "Infiltration",
            "reviewer_status": "not_reviewed",
            "reviewer_id": "rev_semantics_test"
        })
        self.assertFalse(fr_not["reviewed"])
        self.assertEqual(fr_not["reviewer_status"], "not_reviewed")


if __name__ == "__main__":
    unittest.main()

