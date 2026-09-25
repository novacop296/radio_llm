"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.0 — Web API & UI Refinement Test Suite

Verifies:
1. GET /api/studies returns list of studies.
2. GET /api/studies/CXR1122 returns complete aggregated study data.
3. GET /api/studies/CXR1122/image returns valid PNG binary stream.
4. GET /api/studies/CXR1122/evidence returns 7 evidence findings.
5. GET /api/studies/CXR1122/qa returns candidate findings and 14 evaluated questions.
6. GET /api/studies/CXR1122/grounding returns 7 grounding records with layer norm5.
7. GET /api/studies/CXR1122/finding/Infiltration/grounding returns Infiltration record.
8. GET /api/studies/CXR1122/finding/NonExistent/grounding returns 404.
9. GET /api/studies/CXR1122/report returns structured findings and impression.
10. GET /api/studies/CXR1122/validation returns safety validation badges & zero violations.
11. GET /api/studies/CXR1122/export?format=json returns JSON export with research disclaimer.
12. GET /api/studies/CXR1122/export?format=text returns plain text report export with research disclaimers.
13. GET /api/studies/INVALID_ID returns 404.
14. Ground-truth isolation: No 'reference_ground_truth' or 'report_ground_truth' in study detail.
15. Ground-truth isolation: No ground-truth annotations in evidence endpoint.
16. Ground-truth isolation: No ground-truth annotations in report endpoint.
17. Static file serving: GET / returns index.html with 'Model Activation Score' & visual attribution warnings.
18. Static file serving: GET /styles.css returns CSS stylesheet.
19. Static file serving: GET /app.js returns JavaScript application with semantic status interpretations.
20. Artifact serving: GET /api/artifacts/... serves heatmap/overlay images.
21. CORS headers: Verifies Access-Control-Allow-Origin header is present.
22. Model Activation Terminology: Verifies numeric scores are preserved as raw floats and not converted to percentage strings in JSON.
23. Pipeline Status Integrity: Verifies 6 pipeline stages are present in validation response.
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


class TestPhase10API(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Start test HTTP server on an available localhost port
        cls.port = 8766
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

    def test_01_get_studies_list(self):
        """TEST 1: GET /api/studies returns 200 with study list."""
        status, headers, body = self._get("/api/studies")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("studies", data)
        self.assertGreaterEqual(len(data["studies"]), 1)
        self.assertEqual(data["studies"][0]["study_id"], "CXR1122")

    def test_02_get_study_detail(self):
        """TEST 2: GET /api/studies/CXR1122 returns complete aggregated study data."""
        status, headers, body = self._get("/api/studies/CXR1122")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["study_id"], "CXR1122")
        self.assertIn("evidence", data)
        self.assertIn("groundings", data)
        self.assertIn("report", data)
        self.assertIn("validation", data)
        self.assertIn("disclaimer", data)

    def test_03_get_study_image(self):
        """TEST 3: GET /api/studies/CXR1122/image returns valid PNG binary stream."""
        status, headers, body = self._get("/api/studies/CXR1122/image")
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("Content-Type"), "image/png")
        self.assertGreater(len(body), 1000)

    def test_04_get_study_evidence(self):
        """TEST 4: GET /api/studies/CXR1122/evidence returns 7 evidence findings."""
        status, headers, body = self._get("/api/studies/CXR1122/evidence")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(len(data["evidence"]), 7)
        findings = [e["finding"] for e in data["evidence"]]
        self.assertIn("Infiltration", findings)
        self.assertIn("Pneumothorax", findings)

    def test_05_get_study_qa(self):
        """TEST 5: GET /api/studies/CXR1122/qa returns candidate findings and evaluated questions."""
        status, headers, body = self._get("/api/studies/CXR1122/qa")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("candidates", data)
        self.assertIn("questions", data)
        self.assertEqual(len(data["questions"]), 14)

    def test_06_get_study_grounding(self):
        """TEST 6: GET /api/studies/CXR1122/grounding returns 7 grounding records with layer norm5."""
        status, headers, body = self._get("/api/studies/CXR1122/grounding")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(len(data["groundings"]), 7)
        for g in data["groundings"]:
            self.assertEqual(g["target_layer"], "model.features.norm5")
            self.assertIn("heatmap_url", g)
            self.assertIn("overlay_url", g)

    def test_07_get_specific_finding_grounding(self):
        """TEST 7: GET /api/studies/CXR1122/finding/Infiltration/grounding returns Infiltration record."""
        status, headers, body = self._get("/api/studies/CXR1122/finding/Infiltration/grounding")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["finding"], "Infiltration")
        self.assertEqual(data["model_score"], 0.475)

    def test_08_get_specific_finding_not_found(self):
        """TEST 8: GET /api/studies/CXR1122/finding/FakeFinding/grounding returns 404."""
        status, headers, body = self._get("/api/studies/CXR1122/finding/FakeFinding/grounding")
        self.assertEqual(status, 404)

    def test_09_get_study_report(self):
        """TEST 9: GET /api/studies/CXR1122/report returns structured findings and impression."""
        status, headers, body = self._get("/api/studies/CXR1122/report")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("findings", data)
        self.assertIn("impression", data)
        self.assertEqual(len(data["findings"]), 7)

    def test_10_get_study_validation(self):
        """TEST 10: GET /api/studies/CXR1122/validation returns safety validation badges & 0 violations."""
        status, headers, body = self._get("/api/studies/CXR1122/validation")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["schema_validation"], "PASS")
        self.assertEqual(data["ground_truth_isolation"], "PASS")
        self.assertEqual(data["safety_violations_count"], 0)

    def test_11_export_json(self):
        """TEST 11: GET /api/studies/CXR1122/export?format=json returns JSON export with research disclaimer."""
        status, headers, body = self._get("/api/studies/CXR1122/export?format=json")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("disclaimer", data)
        self.assertIn("report", data)
        self.assertEqual(data["study_id"], "CXR1122")

    def test_12_export_text(self):
        """TEST 12: GET /api/studies/CXR1122/export?format=text returns plain text report export."""
        status, headers, body = self._get("/api/studies/CXR1122/export?format=text")
        self.assertEqual(status, 200)
        text = body.decode("utf-8")
        self.assertIn("EXPLAINABLE RADIOLOGY REPORT", text)
        self.assertIn("FINDINGS:", text)
        self.assertIn("IMPRESSION:", text)
        self.assertIn("RESEARCH PROTOTYPE", text)
        self.assertIn("REQUIRES HUMAN REVIEW", text)

    def test_13_invalid_study_handling(self):
        """TEST 13: GET /api/studies/NON_EXISTENT_STUDY returns 404."""
        status, headers, body = self._get("/api/studies/NON_EXISTENT_STUDY")
        self.assertEqual(status, 404)

    def test_14_ground_truth_isolation_in_study_detail(self):
        """TEST 14: Confirms zero ground-truth leakage in aggregated study response."""
        status, headers, body = self._get("/api/studies/CXR1122")
        text = body.decode("utf-8")
        self.assertNotIn("reference_ground_truth", text)
        self.assertNotIn("report_ground_truth", text)
        self.assertNotIn("ground_truth_findings", text)

    def test_15_ground_truth_isolation_in_evidence(self):
        """TEST 15: Confirms zero ground-truth leakage in evidence endpoint."""
        status, headers, body = self._get("/api/studies/CXR1122/evidence")
        text = body.decode("utf-8")
        self.assertNotIn("report_ground_truth", text)
        self.assertNotIn("matched_text", text)

    def test_16_ground_truth_isolation_in_report(self):
        """TEST 16: Confirms zero ground-truth leakage in report endpoint."""
        status, headers, body = self._get("/api/studies/CXR1122/report")
        text = body.decode("utf-8")
        self.assertNotIn("ground_truth", text)

    def test_17_static_frontend_serving(self):
        """TEST 17: GET / returns HTML content with safety warnings and model activation score."""
        status, headers, body = self._get("/")
        self.assertEqual(status, 200)
        text = body.decode("utf-8")
        self.assertIn("<!DOCTYPE html>", text)
        self.assertIn("Explainable Radiology", text)
        self.assertIn("Model Activation Score", text)
        self.assertIn("VISUAL ATTRIBUTION — NOT LESION LOCALIZATION", text)
        self.assertIn("RESEARCH OUTPUT — REQUIRES HUMAN REVIEW", text)

    def test_18_static_css_and_js_serving(self):
        """TEST 18: GET /styles.css and GET /app.js return correct assets."""
        status_css, headers_css, body_css = self._get("/styles.css")
        self.assertEqual(status_css, 200)
        self.assertIn("text/css", headers_css.get("Content-Type", ""))

        status_js, headers_js, body_js = self._get("/app.js")
        self.assertEqual(status_js, 200)
        self.assertIn("javascript", headers_js.get("Content-Type", ""))

    def test_19_artifact_heatmap_serving(self):
        """TEST 19: GET /api/artifacts/... serves heatmap images."""
        status, headers, body = self._get("/api/artifacts/grounding/CXR1122_CXR1122_IM-0080-1001-0002_infiltration_heatmap.png")
        self.assertEqual(status, 200)
        self.assertGreater(len(body), 500)

    def test_20_cors_headers_present(self):
        """TEST 20: Verifies Access-Control-Allow-Origin header is present."""
        status, headers, body = self._get("/api/studies")
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "*")

    def test_21_model_activation_score_numeric(self):
        """TEST 21: Model activation score is numeric float and never a percentage string."""
        status, headers, body = self._get("/api/studies/CXR1122/evidence")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        for ev in data["evidence"]:
            self.assertIsInstance(ev["model_score"], float)
            self.assertGreaterEqual(ev["model_score"], 0.0)
            self.assertLessEqual(ev["model_score"], 1.0)

    def test_22_pipeline_stages_present(self):
        """TEST 22: Validation payload contains 6 pipeline stages."""
        status, headers, body = self._get("/api/studies/CXR1122/validation")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("pipeline_stages", data)
        stages = data["pipeline_stages"]
        for s in ["vision", "qa", "evidence", "grounding", "llm", "validation"]:
            self.assertIn(s, stages)
            self.assertTrue(stages[s])

    def test_23_top_level_tab_navigation_elements(self):
        """TEST 23: Confirms all 9 top-level tabs and corresponding workspace containers exist in frontend."""
        status, headers, body = self._get("/")
        self.assertEqual(status, 200)
        html = body.decode("utf-8")

        expected_tabs = [
            ("tabDatasetQueue", "datasetQueueViewContainer"),
            ("tabIndividualReview", "individualViewContainer"),
            ("tabConsensusDashboard", "consensusDashboardContainer"),
            ("tabEvaluationAnalytics", "evaluationAnalyticsContainer"),
            ("tabResearchEvaluation", "researchEvaluationViewContainer"),
            ("tabExperimentRegistry", "experimentRegistryViewContainer"),
            ("tabExternalBenchmarking", "externalBenchmarkingViewContainer"),
            ("tabResearchExperiments", "researchExperimentsContainer"),
            ("tabCounterfactualExplainability", "counterfactualViewContainer")
        ]

        for tab_id, container_id in expected_tabs:
            self.assertIn(f'id="{tab_id}"', html, f"Tab button id '{tab_id}' missing from index.html")
            self.assertIn(f'id="{container_id}"', html, f"Container id '{container_id}' missing from index.html")

        # Verify app.js serves valid JS containing centralized tab navigation
        status_js, _, body_js = self._get("/app.js")
        self.assertEqual(status_js, 200)
        js = body_js.decode("utf-8")
        self.assertIn("const TABS = {", js)
        self.assertIn("function switchTab(", js)


if __name__ == "__main__":
    unittest.main()
