"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.1 — Multi-Study Browsing, Dual-View Radiograph Support, Reviewer Annotations,
and Batch Processing Test Suite

Verifies:
1. Study index generation (GET /api/studies)
2. Study lookup (GET /api/studies/CXR1122)
3. Multiple image/view discovery (GET /api/studies/{study_id}/images)
4. Single-view fallback
5. Dual-view metadata structure
6. Missing-view handling
7. Grounding availability handling
8. Reviewer annotation creation (POST /api/studies/{study_id}/reviews)
9. Reviewer annotation retrieval (GET /api/studies/{study_id}/reviews)
10. Reviewer annotation deletion (DELETE /api/studies/{study_id}/reviews/{finding})
11. Invalid annotation rejection (malformed payload or invalid status)
12. Machine evidence immutability (original evidence unaffected by reviewer annotations)
13. Ground-truth leakage protection (no ground truth in responses)
14. Batch request validation (POST /api/batch/run rejection of invalid payloads)
15. Batch status handling (GET /api/batch/status returns active job/status)
16. Existing API backward compatibility (GET /evidence, /qa, /grounding, /report, /validation)
17. Export compatibility (GET /export?format=json and format=text)
18. No fabricated heatmaps (missing groundings are empty/omitted, not fake)
19. No fabricated anatomical locations (unspecified is preserved)
20. No activation-to-probability conversion (raw floats preserved)
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

from api import RadiologyAPIHandler, discover_available_studies, load_study_data
from batch_processor import BatchProcessor


class TestPhase11MultiStudy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Start test HTTP server on an available localhost port
        cls.port = 8767
        cls.server = HTTPServer(("127.0.0.1", cls.port), RadiologyAPIHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

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

    def _delete(self, path: str):
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, method="DELETE")
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    def test_01_study_index_generation(self):
        """TEST 1: Study index generation via GET /api/studies."""
        status, headers, body = self._get("/api/studies")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("studies", data)
        self.assertIn("total", data)
        self.assertGreaterEqual(data["total"], 1)
        
        # Verify schema of study index items
        for s in data["studies"]:
            self.assertIn("study_id", s)
            self.assertIn("images_count", s)
            self.assertIn("views", s)
            self.assertIn("report_available", s)
            self.assertIn("evidence_available", s)
            self.assertIn("grounding_available", s)
            self.assertIn("validation_status", s)

    def test_02_study_lookup(self):
        """TEST 2: Study lookup for CXR1122 returns expected aggregated bundle."""
        status, headers, body = self._get("/api/studies/CXR1122")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["study_id"], "CXR1122")
        self.assertIn("evidence", data)
        self.assertIn("groundings", data)
        self.assertIn("report", data)

    def test_03_multiple_image_view_discovery(self):
        """TEST 3: GET /api/studies/CXR1122/images returns available images and views."""
        status, headers, body = self._get("/api/studies/CXR1122/images")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["study_id"], "CXR1122")
        self.assertIn("images", data)
        self.assertGreaterEqual(len(data["images"]), 1)
        
        primary_img = data["images"][0]
        self.assertIn("image_id", primary_img)
        self.assertIn("view", primary_img)
        self.assertTrue(primary_img["available"])

    def test_04_single_view_fallback(self):
        """TEST 4: Non-existent secondary image returns 404 cleanly."""
        status, headers, body = self._get("/api/studies/CXR1122/image/non_existent_image_id")
        self.assertEqual(status, 404)

    def test_05_dual_view_metadata_structure(self):
        """TEST 5: Dual-view studies discovery correctly tags multiple views if present."""
        studies = discover_available_studies()
        self.assertIsInstance(studies, list)
        cxr1122 = next((s for s in studies if s["study_id"] == "CXR1122"), None)
        self.assertIsNotNone(cxr1122)
        self.assertIn("views", cxr1122)
        self.assertIn("Frontal", cxr1122["views"])

    def test_06_missing_view_handling(self):
        """TEST 6: Querying an invalid study ID for images returns 404."""
        status, headers, body = self._get("/api/studies/CXR999999/images")
        self.assertEqual(status, 404)

    def test_07_grounding_availability_handling(self):
        """TEST 7: Grounding metadata clearly specifies available findings and norm5 layer."""
        status, headers, body = self._get("/api/studies/CXR1122/grounding")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("groundings", data)
        findings = [g["finding"] for g in data["groundings"]]
        self.assertIn("Infiltration", findings)
        self.assertIn("Effusion", findings)

    def test_08_reviewer_annotation_creation(self):
        """TEST 8: POST /api/studies/CXR1122/reviews creates a research reviewer annotation."""
        payload = {
            "study_id": "CXR1122",
            "image_id": "CXR1122_IM-0080-1001-0002",
            "finding": "Infiltration",
            "reviewer_status": "confirmed_present",
            "location": "right_lower_lobe",
            "severity": "mild",
            "notes": "Exploratory research review annotation test."
        }
        status, headers, body = self._post("/api/studies/CXR1122/reviews", payload)
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertTrue(data.get("saved"))
        self.assertEqual(data.get("finding"), "Infiltration")

    def test_09_reviewer_annotation_retrieval(self):
        """TEST 9: GET /api/studies/CXR1122/reviews retrieves saved review."""
        status, headers, body = self._get("/api/studies/CXR1122/reviews")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("reviews", data)
        self.assertIn("Infiltration", data["reviews"])
        rev = data["reviews"]["Infiltration"]
        self.assertEqual(rev["reviewer_status"], "confirmed_present")
        self.assertEqual(rev["location"], "right_lower_lobe")
        self.assertEqual(rev["severity"], "mild")
        self.assertIn("disclaimer", data)

    def test_10_reviewer_annotation_deletion(self):
        """TEST 10: DELETE /api/studies/CXR1122/reviews/Infiltration removes annotation."""
        status, headers, body = self._delete("/api/studies/CXR1122/reviews/Infiltration")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertTrue(data.get("deleted"))

        # Verify it's no longer present
        status2, _, body2 = self._get("/api/studies/CXR1122/reviews")
        data2 = json.loads(body2.decode("utf-8"))
        self.assertNotIn("Infiltration", data2.get("reviews", {}))

    def test_11_invalid_annotation_rejection(self):
        """TEST 11: Invalid review payloads are strictly rejected with 400."""
        # Missing finding
        bad_payload_1 = {"reviewer_status": "confirmed_present"}
        status1, _, _ = self._post("/api/studies/CXR1122/reviews", bad_payload_1)
        self.assertEqual(status1, 400)

        # Invalid reviewer_status value
        bad_payload_2 = {"finding": "Pneumothorax", "reviewer_status": "DEFINITIVELY_DIAGNOSED_TRUTH"}
        status2, _, _ = self._post("/api/studies/CXR1122/reviews", bad_payload_2)
        self.assertEqual(status2, 400)

    def test_12_machine_evidence_immutability(self):
        """TEST 12: Machine evidence remains completely immutable after reviewer actions."""
        # 1. Check original machine status
        _, _, body_orig = self._get("/api/studies/CXR1122/evidence")
        data_orig = json.loads(body_orig.decode("utf-8"))
        infil_orig = next(e for e in data_orig["evidence"] if e["finding"] == "Infiltration")
        orig_status = infil_orig["status"]
        orig_score = infil_orig["model_score"]

        # 2. Add reviewer annotation saying confirmed_absent
        rev_payload = {
            "study_id": "CXR1122",
            "finding": "Infiltration",
            "reviewer_status": "confirmed_absent",
            "notes": "Reviewer disagrees with model candidate."
        }
        self._post("/api/studies/CXR1122/reviews", rev_payload)

        # 3. Check machine evidence again
        _, _, body_post = self._get("/api/studies/CXR1122/evidence")
        data_post = json.loads(body_post.decode("utf-8"))
        infil_post = next(e for e in data_post["evidence"] if e["finding"] == "Infiltration")

        self.assertEqual(infil_post["status"], orig_status)
        self.assertEqual(infil_post["model_score"], orig_score)

        # Cleanup
        self._delete("/api/studies/CXR1122/reviews/Infiltration")

    def test_13_ground_truth_leakage_protection(self):
        """TEST 13: Ground-truth reference reports and annotations are never exposed in any API."""
        status, _, body = self._get("/api/studies/CXR1122")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertNotIn("reference_ground_truth", data)
        self.assertNotIn("report_ground_truth", data)
        self.assertNotIn("xml_report", data)

    def test_14_batch_request_validation(self):
        """TEST 14: POST /api/batch/run rejects invalid empty or malformed payloads."""
        # Missing study_ids
        bad_req = {"invalid_field": 123}
        status1, _, body1 = self._post("/api/batch/run", bad_req)
        self.assertEqual(status1, 400)

        # Non-list study_ids
        bad_req2 = {"study_ids": "CXR1122"}
        status2, _, body2 = self._post("/api/batch/run", bad_req2)
        self.assertEqual(status2, 400)

    def test_15_batch_status_handling(self):
        """TEST 15: POST /api/batch/run triggers job and GET /api/batch/status reports progress."""
        payload = {"study_ids": ["CXR1122"]}
        status, _, body = self._post("/api/batch/run", payload)
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("job_id", data)
        self.assertIn("status", data)

        # Query batch status
        status2, _, body2 = self._get("/api/batch/status")
        self.assertEqual(status2, 200)
        data2 = json.loads(body2.decode("utf-8"))
        self.assertIn("job_id", data2)
        self.assertIn("current_stage", data2)
        self.assertIn("results", data2)

    def test_16_existing_api_backward_compatibility(self):
        """TEST 16: All Phase 1.0 endpoints continue to function without changes."""
        endpoints = [
            "/api/studies",
            "/api/studies/CXR1122",
            "/api/studies/CXR1122/image",
            "/api/studies/CXR1122/evidence",
            "/api/studies/CXR1122/qa",
            "/api/studies/CXR1122/grounding",
            "/api/studies/CXR1122/finding/Infiltration/grounding",
            "/api/studies/CXR1122/report",
            "/api/studies/CXR1122/validation",
            "/api/studies/CXR1122/export"
        ]
        for ep in endpoints:
            status, _, body = self._get(ep)
            self.assertEqual(status, 200, f"Endpoint {ep} failed with status {status}")

    def test_17_export_compatibility(self):
        """TEST 17: JSON and plain text exports are valid and contain disclaimers."""
        # JSON export
        s1, _, b1 = self._get("/api/studies/CXR1122/export?format=json")
        self.assertEqual(s1, 200)
        j_exp = json.loads(b1.decode("utf-8"))
        self.assertIn("research_disclaimer", j_exp)

        # Text export
        s2, _, b2 = self._get("/api/studies/CXR1122/export?format=text")
        self.assertEqual(s2, 200)
        t_exp = b2.decode("utf-8")
        self.assertIn("RESEARCH PROTOTYPE — NOT FOR CLINICAL USE", t_exp)

    def test_18_no_fabricated_heatmaps(self):
        """TEST 18: Querying non-existent finding grounding returns 404 without fabrication."""
        status, _, _ = self._get("/api/studies/CXR1122/finding/FakePathology/grounding")
        self.assertEqual(status, 404)

    def test_19_no_fabricated_anatomical_locations(self):
        """TEST 19: Unspecified locations remain 'unspecified' and are not fabricated."""
        status, _, body = self._get("/api/studies/CXR1122/evidence")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        for ev in data["evidence"]:
            if ev["location"] != "unspecified":
                self.assertIn(ev["location"], ["lungs", "right_lower_lobe", "left_lower_lobe", "bilateral_bases", "unspecified"])

    def test_20_no_activation_to_probability_conversion(self):
        """TEST 20: Model activation scores are preserved as raw floats and not converted to percentage strings."""
        status, _, body = self._get("/api/studies/CXR1122/evidence")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        for ev in data["evidence"]:
            self.assertIsInstance(ev["model_score"], (float, int))
            self.assertNotIn("%", str(ev["model_score"]))


if __name__ == "__main__":
    unittest.main()
