"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.5 — 20-Stage End-to-End Research Evaluation Verification Pipeline

Stages:
 1. Verify storage isolation and record SHA-256 checksums of data/iu_xray/.
 2. Create immutable dataset snapshot over study cohort.
 3. Verify snapshot manifest hash reproducibility and disk persistence.
 4. Verify snapshot cryptographic integrity validator.
 5. Create Experiment 1 (DenseNet-121, norm5, threshold 0.15).
 6. Verify deterministic SHA-256 configuration fingerprint (zero secrets).
 7. Execute Experiment 1 and verify transition to COMPLETED.
 8. Verify complete 11-stage provenance audit chain.
 9. Verify research evaluation metrics (bounds [-1,1], [0,1], no NaN/Inf).
10. Create Experiment 2 with alternate configuration (denseblock4, threshold 0.25).
11. Execute Experiment 2 and verify completed state.
12. Perform side-by-side comparison across Experiment 1 and Experiment 2.
13. Verify configuration difference detection.
14. Verify metric comparison matrices and non-evaluative terminology.
15. Validate experiment records with ExperimentValidator (all invariants pass).
16. Enforce completed experiment immutability (mutation attempts rejected).
17. Verify experiment archival lifecycle transition.
18. Generate and verify sanitized JSON and plain-text export packages.
19. Verify strict ground-truth report leakage isolation across all outputs.
20. Confirm data/iu_xray/ byte immutability (SHA256_BEFORE == SHA256_AFTER).
"""

import os
import sys
import json
import time
import math
import hashlib
from typing import Dict, List, Any, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from dataset_snapshot_manager import global_snapshot_manager, DatasetSnapshotManager
from experiment_manager import global_experiment_manager, ExperimentManager
from experiment_runner import global_experiment_runner, ExperimentRunner
from experiment_comparator import global_experiment_comparator, ExperimentComparator
from research_metrics import global_research_metrics_calculator
from validate_experiment import global_experiment_validator

DATA_DIR = os.path.join(BASE_DIR, "data", "iu_xray")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "data", "snapshots")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "data", "experiments")


def calculate_dir_sha256(directory_path: str) -> Dict[str, str]:
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


def print_step(num: int, title: str):
    print(f"\n[{num:02d}/20] {title}")


def run_e2e_verification():
    print("=" * 70)
    print("PHASE 1.5 — 20-STAGE END-TO-END RESEARCH EVALUATION PIPELINE")
    print("RESEARCH EVALUATION ONLY — NOT CLINICAL PERFORMANCE")
    print("=" * 70)

    # -------------------------------------------------------------
    # Stage 1: Verify data isolation and record baseline SHA-256
    # -------------------------------------------------------------
    print_step(1, "Verifying Storage Isolation & Recording Pre-Run Hashes")
    pre_run_hashes = calculate_dir_sha256(DATA_DIR)
    print(f"✓ Recorded baseline SHA-256 hashes for {len(pre_run_hashes)} files in data/iu_xray/")
    assert len(pre_run_hashes) > 0, "Machine evidence directory must contain files"

    # -------------------------------------------------------------
    # Stage 2: Create Immutable Dataset Snapshot
    # -------------------------------------------------------------
    print_step(2, "Creating Immutable Dataset Snapshot")
    snap_1 = global_snapshot_manager.create_snapshot(
        name="Phase 1.5 Benchmark Snapshot",
        description="Standard cohort for Phase 1.5 evaluation",
        study_ids=["CXR1122"],
        created_by="e2e_evaluator"
    )
    snap_id = snap_1["snapshot_id"]
    print(f"✓ Snapshot created: '{snap_id}' ({snap_1['study_count']} studies)")
    print(f"  Manifest SHA-256: {snap_1['manifest_sha256']}")
    assert snap_1["study_count"] == 1
    assert len(snap_1["manifest_sha256"]) == 64

    # -------------------------------------------------------------
    # Stage 3: Verify Manifest Hash Reproducibility
    # -------------------------------------------------------------
    print_step(3, "Verifying Snapshot Manifest Hash Reproducibility")
    snap_repro = global_snapshot_manager.create_snapshot(
        name="Reproducibility Check",
        description="Duplicate study list",
        study_ids=["CXR1122"]
    )
    print(f"✓ Reproducibility Hash: {snap_repro['manifest_sha256']}")
    assert snap_1["manifest_sha256"] == snap_repro["manifest_sha256"], "Manifest hash must be deterministic"

    # -------------------------------------------------------------
    # Stage 4: Snapshot Integrity Validation
    # -------------------------------------------------------------
    print_step(4, "Validating Snapshot Cryptographic Integrity")
    is_valid, errors = global_snapshot_manager.validate_snapshot_integrity(snap_id)
    assert is_valid, f"Snapshot integrity failed: {errors}"
    print(f"✓ Snapshot '{snap_id}' passed cryptographic integrity verification (0 errors)")

    # -------------------------------------------------------------
    # Stage 5: Create Experiment 1
    # -------------------------------------------------------------
    print_step(5, "Creating Research Experiment 1 (Norm5 Baseline)")
    exp_1 = global_experiment_manager.create_experiment(
        name="DenseNet121_norm5_Baseline",
        description="Baseline evaluation with DenseNet-121 layer norm5",
        dataset_snapshot_id=snap_id,
        configuration={
            "model_name": "TorchXRayVision DenseNet-121",
            "model_version": "densenet121-res224-all",
            "target_layer": "model.features.norm5",
            "qa_threshold": 0.15,
            "qa_top_k": 5
        },
        created_by="researcher_01"
    )
    exp1_id = exp_1["experiment_id"]
    print(f"✓ Experiment 1 created: '{exp1_id}' (Status: {exp_1['status']})")
    assert exp_1["status"] == "CREATED"

    # -------------------------------------------------------------
    # Stage 6: Verify Configuration Fingerprint
    # -------------------------------------------------------------
    print_step(6, "Verifying Deterministic SHA-256 Configuration Fingerprint")
    fp1 = exp_1["configuration_fingerprint"]
    print(f"✓ Fingerprint SHA-256: {fp1['fingerprint_sha256']}")
    assert len(fp1["fingerprint_sha256"]) == 64
    assert "api_key" not in json.dumps(fp1)

    # -------------------------------------------------------------
    # Stage 7: Execute Experiment 1
    # -------------------------------------------------------------
    print_step(7, "Executing Experiment 1 Across Snapshot")
    res1 = global_experiment_runner.run_experiment(exp1_id)
    print(f"✓ Experiment 1 executed successfully! Status: {res1['status']}")
    assert res1["status"] == "COMPLETED"
    assert res1["completed_at"] is not None

    # -------------------------------------------------------------
    # Stage 8: Verify 11-Stage Provenance Audit Chain
    # -------------------------------------------------------------
    print_step(8, "Verifying Complete 11-Stage Provenance Chain")
    prov1 = res1.get("provenance", {})
    stages1 = prov1.get("pipeline_stages", [])
    print(f"✓ Provenance pipeline stages count: {len(stages1)}")
    assert len(stages1) == 11, f"Expected 11 pipeline stages, got {len(stages1)}"
    for idx, stg in enumerate(stages1, 1):
        print(f"   [{idx:02d}] {stg['stage_name']} -> {stg['status']}")
        assert stg["status"] == "COMPLETED"

    # -------------------------------------------------------------
    # Stage 9: Verify Metrics Bounds and Absence of NaN/Infinity
    # -------------------------------------------------------------
    print_step(9, "Verifying Metric Mathematical Bounds & Absence of NaN/Infinity")
    metrics1 = res1.get("metrics", {})
    metrics_str = json.dumps(metrics1)
    assert "NaN" not in metrics_str
    assert "Infinity" not in metrics_str

    rev_cov = metrics1.get("review_coverage", {})
    assert 0.0 <= rev_cov.get("review_completion_ratio", 0.0) <= 1.0
    print(f"✓ Review coverage completion ratio: {rev_cov.get('review_completion_ratio')}")

    irr = metrics1.get("inter_rater_reliability", {})
    ck = irr.get("cohens_kappa_2_reviewers", {}).get("average_kappa")
    fk = irr.get("fleiss_kappa_multi_reviewers", {}).get("average_kappa")
    if ck is not None:
        assert -1.0 <= ck <= 1.0
    if fk is not None:
        assert -1.0 <= fk <= 1.0
    print(f"✓ Inter-rater reliability: Cohen κ={ck}, Fleiss κ={fk}")

    # -------------------------------------------------------------
    # Stage 10: Create Experiment 2 (Alternate Layer Configuration)
    # -------------------------------------------------------------
    print_step(10, "Creating Research Experiment 2 (DenseBlock4 Alternate)")
    exp_2 = global_experiment_manager.create_experiment(
        name="DenseNet121_denseblock4_Alternate",
        description="Alternate evaluation targeting DenseBlock4",
        dataset_snapshot_id=snap_id,
        configuration={
            "model_name": "TorchXRayVision DenseNet-121",
            "model_version": "densenet121-res224-all",
            "target_layer": "model.features.denseblock4",
            "qa_threshold": 0.25,
            "qa_top_k": 5
        },
        created_by="researcher_02"
    )
    exp2_id = exp_2["experiment_id"]
    print(f"✓ Experiment 2 created: '{exp2_id}'")

    # -------------------------------------------------------------
    # Stage 11: Execute Experiment 2
    # -------------------------------------------------------------
    print_step(11, "Executing Experiment 2")
    res2 = global_experiment_runner.run_experiment(exp2_id)
    assert res2["status"] == "COMPLETED"
    print(f"✓ Experiment 2 execution COMPLETED")

    # -------------------------------------------------------------
    # Stage 12: Perform Side-by-Side Comparison
    # -------------------------------------------------------------
    print_step(12, "Performing Side-by-Side Experiment Comparison")
    comp = global_experiment_comparator.compare_experiments([exp1_id, exp2_id])
    assert len(comp["experiment_ids"]) == 2
    print(f"✓ Successfully generated side-by-side comparison for [{exp1_id}, {exp2_id}]")

    # -------------------------------------------------------------
    # Stage 13: Verify Configuration Difference Detection
    # -------------------------------------------------------------
    print_step(13, "Verifying Configuration Difference Detection")
    cfg_diffs = {d["configuration_field"]: d for d in comp.get("configuration_differences", [])}
    assert "target_layer" in cfg_diffs, "Target layer difference must be detected"
    assert "qa_threshold" in cfg_diffs, "QA threshold difference must be detected"
    print(f"✓ Detected {len(cfg_diffs)} configuration differences:")
    for field, item in cfg_diffs.items():
        print(f"   - {field}: {item['values_by_experiment']}")

    # -------------------------------------------------------------
    # Stage 14: Verify Metric Comparison Matrices & Terminology
    # -------------------------------------------------------------
    print_step(14, "Verifying Metric Comparison & Non-Evaluative Terminology")
    m_comp = comp.get("metric_comparisons", {})
    assert "review_coverage" in m_comp
    assert "machine_reviewer_agreement" in m_comp
    comp_json_lower = json.dumps(comp).lower()
    for forbidden in ["winner", "superior", "rank", "best", "clinical trial"]:
        assert forbidden not in comp_json_lower, f"Forbidden evaluative term found: '{forbidden}'"
    print("✓ All comparison terminology verified as strictly non-evaluative and objective.")

    # -------------------------------------------------------------
    # Stage 15: Validate Experiment Record Invariants
    # -------------------------------------------------------------
    print_step(15, "Validating Experiment Invariants with ExperimentValidator")
    report1 = global_experiment_validator.validate(exp1_id)
    assert report1["is_valid"], f"Experiment 1 failed validation: {report1['violations']}"
    report2 = global_experiment_validator.validate(exp2_id)
    assert report2["is_valid"], f"Experiment 2 failed validation: {report2['violations']}"
    print("✓ Both experiments passed 100% of invariant security and integrity checks.")

    # -------------------------------------------------------------
    # Stage 16: Enforce Completed Experiment Immutability
    # -------------------------------------------------------------
    print_step(16, "Enforcing Completed Experiment Immutability")
    try:
        global_experiment_manager.update_experiment_status(exp1_id, "RUNNING")
        raise AssertionError("Mutation of completed experiment must be blocked!")
    except ValueError as e:
        print(f"✓ Immutability successfully protected: {e}")

    # -------------------------------------------------------------
    # Stage 17: Verify Experiment Archival Lifecycle
    # -------------------------------------------------------------
    print_step(17, "Verifying Experiment Archival Transition")
    archived = global_experiment_manager.archive_experiment(exp2_id, reason="E2E test run archival")
    assert archived["status"] == "ARCHIVED"
    assert archived.get("archive_reason") == "E2E test run archival"
    print(f"✓ Experiment '{exp2_id}' successfully transitioned to ARCHIVED state.")

    # -------------------------------------------------------------
    # Stage 18: Generate and Verify Sanitized Export Packages
    # -------------------------------------------------------------
    print_step(18, "Generating and Verifying JSON/Text Export Packages")
    json_txt, j_mime = global_experiment_runner.export_experiment(exp1_id, format="json")
    assert j_mime == "application/json"
    json_pkg = json.loads(json_txt)
    assert "experiment_id" in json_pkg
    assert "research_disclaimer" in json_pkg

    text_txt, t_mime = global_experiment_runner.export_experiment(exp1_id, format="text")
    assert "text/plain" in t_mime
    assert "RESEARCH EVALUATION ONLY — NOT CLINICAL PERFORMANCE" in text_txt
    print(f"✓ Generated sanitized JSON export ({len(json_txt)} bytes) and Text export ({len(text_txt)} bytes).")

    # -------------------------------------------------------------
    # Stage 19: Strict Ground-Truth Isolation Verification
    # -------------------------------------------------------------
    print_step(19, "Verifying Strict Zero Ground-Truth Leakage Isolation")
    for payload_str in [json_txt, text_txt, json.dumps(comp)]:
        lower_p = payload_str.lower()
        for forbidden in ["<efind>", "<eimpression>", "reference report", "patient_name", "mrn"]:
            assert forbidden not in lower_p, f"Ground truth leaked in payload: '{forbidden}'"
    print("✓ Verified zero IU X-Ray reference reports or clinical ground truth leaked.")

    # -------------------------------------------------------------
    # Stage 20: Post-Run Byte Immutability Check
    # -------------------------------------------------------------
    print_step(20, "Verifying Post-Run SHA-256 Byte Immutability (data/iu_xray/)")
    post_run_hashes = calculate_dir_sha256(DATA_DIR)
    assert pre_run_hashes == post_run_hashes, "Machine evidence files in data/iu_xray/ MUST NOT BE MODIFIED!"
    print(f"✓ 100% Byte Immutability Verified: Pre-Run == Post-Run ({len(post_run_hashes)} files unchanged).")

    print("\n" + "=" * 70)
    print("✅ ALL 20 STAGES OF PHASE 1.5 END-TO-END PIPELINE PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_e2e_verification()
