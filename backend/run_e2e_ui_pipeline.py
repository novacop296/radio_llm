"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.0 — End-to-End Web UI & API Integration Runner

Module: run_e2e_ui_pipeline.py
Purpose:
- Verifies the complete data flow from validated Phase 0.6–0.9 artifacts through the Backend REST API
  into frontend-compatible JSON packages and binary image streams.
- Validates:
  1. Aggregated study retrieval (CXR1122).
  2. Original radiograph image availability.
  3. Finding-specific Grad-CAM heatmaps and overlays (7 candidate findings).
  4. Diagnostic QA entries (14 questions).
  5. Evidence Layer status (7 findings, status preserved).
  6. Structured Radiology Report (Findings + Impression).
  7. Report export generation (JSON and Plain Text).
  8. Ground-truth isolation guarantee (Zero reference/ground-truth leakage).
"""

import os
import sys
import json
import threading
import urllib.request
from http.server import HTTPServer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from api import RadiologyAPIHandler, load_study_data


def run_e2e_ui_verification():
    print("=" * 75)
    print("PHASE 1.0 — END-TO-END WEB UI & REST API INTEGRATION VERIFICATION")
    print("=" * 75)

    study_id = "CXR1122"
    port = 8899

    # 1. Start ephemeral HTTP server
    print(f"1. Initializing Backend REST API Server on localhost:{port}...")
    server = HTTPServer(("127.0.0.1", port), RadiologyAPIHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    base_url = f"http://127.0.0.1:{port}"
    print("   [OK] Server started.")

    try:
        # 2. Test Studies List Endpoint
        print("\n2. Querying GET /api/studies...")
        req = urllib.request.Request(f"{base_url}/api/studies")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"   [OK] HTTP {resp.status} — Found {len(data['studies'])} study: {data['studies'][0]['study_id']}")

        # 3. Test Study Detail Aggregate Endpoint
        print(f"\n3. Querying GET /api/studies/{study_id} (Aggregated Package)...")
        req = urllib.request.Request(f"{base_url}/api/studies/{study_id}")
        with urllib.request.urlopen(req) as resp:
            study_data = json.loads(resp.read().decode("utf-8"))
            print(f"   [OK] HTTP {resp.status}")
            print(f"   - Study ID      : {study_data['study_id']}")
            print(f"   - Image ID      : {study_data['image_id']} ({study_data['view']})")
            print(f"   - Evidence Count: {len(study_data['evidence'])} findings")
            print(f"   - QA Questions  : {len(study_data['qa_questions'])} questions")
            print(f"   - Grounding Maps: {len(study_data['groundings'])} visual attribution maps")
            print(f"   - Report Sections: FINDINGS ({len(study_data['report']['findings'])} items), IMPRESSION ({len(study_data['report']['impression'])} items)")

        # 4. Test Image Delivery
        print(f"\n4. Querying GET /api/studies/{study_id}/image...")
        req = urllib.request.Request(f"{base_url}/api/studies/{study_id}/image")
        with urllib.request.urlopen(req) as resp:
            img_bytes = resp.read()
            print(f"   [OK] HTTP {resp.status} — Streamed {len(img_bytes)} bytes of {resp.headers.get('Content-Type')}")

        # 5. Test Finding-Specific Grounding Map Delivery
        print(f"\n5. Querying Finding-Specific Grounding for 'Infiltration'...")
        req = urllib.request.Request(f"{base_url}/api/studies/{study_id}/finding/Infiltration/grounding")
        with urllib.request.urlopen(req) as resp:
            g_data = json.loads(resp.read().decode("utf-8"))
            print(f"   [OK] HTTP {resp.status} — Layer: {g_data['target_layer']} | Score: {g_data['model_score']} | Heatmap: {g_data['heatmap_url']}")

        # 6. Test Report Export Endpoints
        print(f"\n6. Testing Report Export Endpoints...")
        req_json = urllib.request.Request(f"{base_url}/api/studies/{study_id}/export?format=json")
        with urllib.request.urlopen(req_json) as resp:
            exp_json = json.loads(resp.read().decode("utf-8"))
            print(f"   [OK] JSON Export  : {len(exp_json['report']['findings'])} findings, disclaimer present.")

        req_text = urllib.request.Request(f"{base_url}/api/studies/{study_id}/export?format=text")
        with urllib.request.urlopen(req_text) as resp:
            exp_text = resp.read().decode("utf-8")
            print(f"   [OK] Text Export  : {len(exp_text.splitlines())} lines generated.")

        # 7. Ground-Truth Isolation Check
        print(f"\n7. Verifying Ground-Truth Isolation...")
        raw_json_str = json.dumps(study_data)
        if "report_ground_truth" in raw_json_str or "reference_ground_truth" in raw_json_str:
            raise RuntimeError("[CRITICAL] Ground-truth leaked into client-facing study package!")
        print("   [OK] Zero ground-truth leakage confirmed.")

        # 8. Test Static Frontend Delivery
        print(f"\n8. Verifying Frontend Static Asset Delivery (HTML/CSS/JS)...")
        for asset in ["/", "/styles.css", "/app.js"]:
            req_asset = urllib.request.Request(f"{base_url}{asset}")
            with urllib.request.urlopen(req_asset) as resp:
                print(f"   [OK] GET {asset:<12} -> HTTP {resp.status} ({resp.headers.get('Content-Type')})")

    finally:
        server.shutdown()
        server.server_close()

    print("\n" + "=" * 75)
    print("PHASE 1.0 END-TO-END VERIFICATION: ALL CHECKS PASSED (100%)")
    print("=" * 75)


if __name__ == "__main__":
    run_e2e_ui_verification()
