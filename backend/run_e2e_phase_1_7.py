"""
Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.7 — End-to-End Experiment Registry & Versioning Verification Pipeline

This comprehensive 26-stage script verifies all capabilities of Phase 1.7:
1. Model discovery & registration
2. Model weight cryptographic identity & hash validation
3. Dataset version creation & deterministic manifest hashing
4. Dataset version finalization & cohort isolation
5. Experiment configuration registration & deterministic fingerprinting
6. Experiment execution linking Model Registry, Dataset Versions & Phase 1.6 Evaluation Engine
7. Statistical distribution summaries & 95% bootstrap uncertainty intervals
8. Finding-level confusion matrix & research error analysis
9. Side-by-side longitudinal experiment comparison (non-evaluative metric deltas & compatibility)
10. Immutable snapshot generation & manifest integrity verification
11. Finalized experiment immutability enforcement (mutation rejection)
12. Zero ground-truth leakage verification (<eFind>, <eImpression>, <AbstractText>)
13. Zero private secret / key / token leakage in fingerprints or payloads
14. Complete byte-level immutability for machine artifacts (data/iu_xray/)
15. REST API endpoint suite verification
16. Multi-format export validation (JSON & plain text)
17. Deterministic reproducibility verification from stored metadata

RESEARCH PROTOTYPE ONLY — NOT CLINICAL VALIDATION EVIDENCE.
"""

import os
import sys
import json
import time
import shutil
import hashlib
import tempfile
import threading
from typing import Dict, List, Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from backend.model_registry import ModelRegistry, global_model_registry
from backend.dataset_version_manager import DatasetVersionManager, global_dataset_version_manager
from backend.experiment_registry import ExperimentRegistry, global_experiment_registry, RESEARCH_DISCLAIMER
from backend.experiment_runner import ExperimentRunner, global_experiment_runner
from backend.experiment_comparator import ExperimentComparator, global_experiment_comparator
from backend.experiment_history import ExperimentHistoryTracker, global_experiment_history
from backend.experiment_snapshot import ExperimentSnapshotManager, global_experiment_snapshot_manager
from backend.validate_phase_1_7 import Phase17Validator
from backend.api import create_server


def compute_dir_sha256(directory_path: str) -> Dict[str, str]:
    """Computes SHA-256 hashes for all files in directory."""
    hashes = {}
    if not os.path.exists(directory_path):
        return hashes
    for root, _, files in os.walk(directory_path):
        for f in sorted(files):
            fp = os.path.join(root, f)
            with open(fp, "rb") as fh:
                hashes[os.path.relpath(fp, directory_path)] = hashlib.sha256(fh.read()).hexdigest()
    return hashes


def run_e2e_pipeline() -> bool:
    print("=" * 80)
    print("PHASE 1.7 — COMPREHENSIVE 26-STAGE END-TO-END VERIFICATION PIPELINE")
    print("Research Experiment Registry, Model Versioning & Longitudinal Tracking")
    print("=" * 80)

    test_temp_dir = tempfile.mkdtemp(prefix="e2e_phase17_")
    models_dir = os.path.join(test_temp_dir, "models")
    versions_dir = os.path.join(test_temp_dir, "dataset_versions")
    experiments_dir = os.path.join(test_temp_dir, "experiments")
    snapshots_dir = os.path.join(test_temp_dir, "snapshots")

    # Record initial machine artifact hashes (data/iu_xray/)
    iu_xray_dir = os.path.join(BASE_DIR, "data", "iu_xray")
    iu_hashes_before = compute_dir_sha256(iu_xray_dir)

    try:
        # Initialize isolated registries for E2E run
        model_reg = ModelRegistry(models_dir=models_dir)
        dsv_mgr = DatasetVersionManager(versions_dir=versions_dir)
        hist_tracker = ExperimentHistoryTracker(history_dir=os.path.join(experiments_dir, "history"))
        exp_reg = ExperimentRegistry(
            experiments_dir=experiments_dir,
            model_registry=model_reg,
            dataset_version_manager=dsv_mgr,
            history_tracker=hist_tracker
        )
        snap_mgr = ExperimentSnapshotManager(snapshots_dir=snapshots_dir)
        exp_reg.snapshot_manager = snap_mgr

        exp_runner = ExperimentRunner(
            experiment_registry=exp_reg,
            model_registry=model_reg,
            dataset_version_manager=dsv_mgr,
            experiments_dir=experiments_dir
        )
        exp_comp = ExperimentComparator(
            experiment_registry=exp_reg,
            model_registry=model_reg,
            dataset_version_manager=dsv_mgr
        )

        # STAGE 1: Discover available model configurations
        print("\n[STAGE 1/26] Discovering model architectures...")
        reg_models = model_reg.list_models()
        print(f"  Available registered models in test registry: {len(reg_models)}")

        # STAGE 2: Register DenseNet-121 model version
        print("\n[STAGE 2/26] Registering DenseNet-121 model version...")
        m1 = model_reg.register_model(
            model_id="model_txrv_densenet121_v1",
            model_name="TorchXRayVision DenseNet-121",
            architecture="DenseNet-121",
            framework="torch",
            framework_version="2.0.1",
            weights_identifier="densenet121-res224-all",
            weights_sha256="4a6a5789b65e10d29abf5f9e8a7c2901",
            input_dimensions=[1, 224, 224],
            preprocessing={"resize": [224, 224], "normalization": "txrv_rescale_minmax_to_neg1024_pos1024"},
            target_labels=["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Nodule", "Pneumonia", "Pneumothorax"],
            target_layer="model.features.norm5",
            registered_by="e2e_researcher"
        )
        assert m1["model_id"] == "model_txrv_densenet121_v1"
        assert m1["status"] == "REGISTERED"
        print(f"  Model registered successfully: {m1['model_id']}")

        # STAGE 3: Verify model identity & cryptographic weights hash
        print("\n[STAGE 3/26] Verifying model weight integrity...")
        ver_res = model_reg.verify_model_hash("model_txrv_densenet121_v1")
        assert ver_res["is_valid"] is True
        print(f"  Model weights SHA-256 validated: {ver_res['weights_sha256']}")

        # STAGE 4: Create dataset version
        print("\n[STAGE 4/26] Creating versioned study cohort...")
        dsv1 = dsv_mgr.create_dataset_version(
            dataset_version_id="dsv_iu_xray_e2e_v1",
            source_dataset="IU_XRAY",
            study_ids=["CXR1122", "CXR1123"],
            allowed_reference_annotations=["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Normal"],
            inclusion_rules={"has_frontal_view": True, "min_resolution": [224, 224]},
            exclusion_rules={"exclude_corrupt": True},
            created_by="e2e_researcher"
        )
        assert dsv1["dataset_version_id"] == "dsv_iu_xray_e2e_v1"
        assert dsv1["study_count"] == 2
        print(f"  Dataset version created: {dsv1['dataset_version_id']}, Manifest SHA-256: {dsv1['manifest_sha256'][:16]}...")

        # STAGE 5: Fingerprint dataset version
        print("\n[STAGE 5/26] Extracting dataset version fingerprint...")
        dsv_fp = dsv_mgr.fingerprint_dataset_version("dsv_iu_xray_e2e_v1")
        assert dsv_fp["manifest_sha256"] == dsv1["manifest_sha256"]
        print(f"  Dataset version fingerprint confirmed: {dsv_fp['manifest_sha256'][:16]}...")

        # STAGE 6: Finalize dataset version
        print("\n[STAGE 6/26] Finalizing dataset version (locking immutability)...")
        fin_dsv = dsv_mgr.finalize_dataset_version("dsv_iu_xray_e2e_v1")
        assert fin_dsv["status"] == "FINALIZED"
        print("  Dataset version locked as FINALIZED.")

        # STAGE 7: Create experiment configuration
        print("\n[STAGE 7/26] Creating reproducible experiment configuration...")
        exp1 = exp_reg.create_experiment(
            experiment_id="exp_e2e_phase17_main",
            experiment_name="Phase 1.7 E2E Main Baseline",
            description="Reproducibility verification run",
            model_id="model_txrv_densenet121_v1",
            dataset_version_id="dsv_iu_xray_e2e_v1",
            evaluation_dataset_id="eval_ds_e2e_v1",
            methodology="standard_qa_evidence_evaluation",
            inference_configuration={"threshold": 0.15, "top_k": 5},
            grounding_configuration={"target_layer": "model.features.norm5"},
            report_configuration={"llm_provider": "mock", "llm_model": "mock-radiology-llm"},
            random_seed=42,
            created_by="e2e_runner"
        )
        assert exp1["status"] == "REGISTERED"
        print(f"  Experiment registered: {exp1['experiment_id']}")

        # STAGE 8: Generate deterministic experiment fingerprint
        print("\n[STAGE 8/26] Generating deterministic configuration fingerprint...")
        fp1 = exp1["fingerprint_sha256"]
        assert len(fp1) == 64
        # Verify determinism: recomputing should yield identical hash
        re_fp = exp_reg.generate_experiment_fingerprint(exp1)
        assert re_fp == fp1, "Fingerprint must be 100% deterministic"
        print(f"  Fingerprint (SHA-256): {fp1}")

        # STAGE 9: Execute experiment using Phase 1.6 evaluation engine
        print("\n[STAGE 9/26] Executing experiment pipeline...")
        run_res = exp_runner.run_experiment("exp_e2e_phase17_main")
        assert run_res["status"] == "COMPLETED"
        print(f"  Experiment completed successfully. Status: {run_res['status']}")

        # STAGE 10: Collect evaluation metrics
        print("\n[STAGE 10/26] Collecting research evaluation metrics...")
        metrics = run_res["metrics"]
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        print(f"  Metrics: Accuracy={metrics.get('accuracy')}, Precision={metrics.get('precision')}, Recall={metrics.get('recall')}, F1={metrics.get('f1_score')}")

        # STAGE 11: Generate statistics & bootstrap uncertainty intervals
        print("\n[STAGE 11/26] Computing statistical distributions and bootstrap intervals...")
        stats = run_res["statistics"]
        assert "overall_summary" in stats
        assert "bootstrap_uncertainty_interval" in stats
        print(f"  Statistical Summary: Mean={stats['overall_summary'].get('mean')}, 95% CI={stats['bootstrap_uncertainty_interval']}")

        # STAGE 12: Generate error analysis & confusion matrix
        print("\n[STAGE 12/26] Generating finding-level research error analysis...")
        errs = run_res["error_analysis"]
        assert "confusion_matrix" in errs
        assert "classification_metrics" in errs
        cm = errs["confusion_matrix"]
        print(f"  Confusion Matrix: TP={cm.get('tp')}, TN={cm.get('tn')}, FP={cm.get('fp')}, FN={cm.get('fn')}")

        # STAGE 13: Create experiment snapshot
        print("\n[STAGE 13/26] Generating immutable research snapshot...")
        snap = snap_mgr.create_snapshot(
            experiment_id="exp_e2e_phase17_main",
            experiment_config=run_res,
            model_version=m1,
            dataset_version=fin_dsv,
            metrics=metrics,
            statistics=stats,
            error_analysis=errs,
            provenance=run_res.get("provenance_trail"),
            validation_status="VALIDATED"
        )
        assert snap["snapshot_sha256"] is not None
        print(f"  Snapshot created: {snap['snapshot_id']}, SHA-256: {snap['snapshot_sha256'][:16]}...")

        # STAGE 14: Validate experiment invariants
        print("\n[STAGE 14/26] Validating experiment schema & research invariants...")
        val_res = exp_reg.validate_experiment("exp_e2e_phase17_main")
        assert val_res["valid"] is True
        print(f"  Validation status: {val_res['status']} (Valid: {val_res['valid']})")

        # STAGE 15: Compare experiment against a variant configuration
        print("\n[STAGE 15/26] Running side-by-side longitudinal experiment comparison...")
        exp2 = exp_reg.create_experiment(
            experiment_id="exp_e2e_phase17_variant",
            experiment_name="Phase 1.7 E2E Variant Layer",
            model_id="model_txrv_densenet121_v1",
            dataset_version_id="dsv_iu_xray_e2e_v1",
            evaluation_dataset_id="eval_ds_e2e_v1",
            methodology="standard_qa_evidence_evaluation",
            inference_configuration={"threshold": 0.25, "top_k": 3},
            grounding_configuration={"target_layer": "model.features.denseblock4"},
            created_by="e2e_runner"
        )
        exp_runner.run_experiment("exp_e2e_phase17_variant")

        comp = exp_comp.compare_experiments(["exp_e2e_phase17_main", "exp_e2e_phase17_variant"])
        assert "configuration_differences" in comp
        assert "metric_comparisons" in comp
        assert "metric_deltas" in comp
        assert "relative_deltas" in comp
        assert "winner" not in json.dumps(comp).lower()
        assert "best" not in json.dumps(comp).lower()
        print(f"  Comparison generated: {len(comp['configuration_differences'])} config differences detected.")
        print(f"  Neutral terminology enforced: True")

        # STAGE 16: Generate experiment provenance
        print("\n[STAGE 16/26] Generating sanitized experiment provenance trail...")
        prov = exp_reg.get_experiment_provenance("exp_e2e_phase17_main")
        assert len(prov) >= 3
        print(f"  Provenance stages logged: {len(prov)} events.")

        # STAGE 17: Finalize experiment
        print("\n[STAGE 17/26] Finalizing experiment (permanent lock)...")
        fin_exp = exp_reg.finalize_experiment("exp_e2e_phase17_main")
        assert fin_exp["status"] == "FINALIZED"
        print(f"  Experiment finalized. Snapshot SHA-256: {fin_exp.get('snapshot_sha256')[:16]}...")

        # STAGE 18: Verify finalized immutability
        print("\n[STAGE 18/26] Verifying immutable state lock on finalized experiment...")
        try:
            exp_reg.update_experiment("exp_e2e_phase17_main", {"description": "Attempted illegal modification"})
            assert False, "Finalized experiment must reject modification"
        except ValueError as ve:
            assert "FINALIZED" in str(ve)
            print("  Immutability confirmed: Mutation rejected with ValueError.")

        # STAGE 19: Verify zero ground-truth report leakage
        print("\n[STAGE 19/26] Auditing for zero ground-truth XML report leakage...")
        for name, payload in [
            ("experiment_record", fin_exp),
            ("comparison_payload", comp),
            ("snapshot_manifest", snap),
            ("provenance_trail", prov)
        ]:
            leak_ok, leak_issues = Phase17Validator.verify_zero_ground_truth_leakage(payload)
            assert leak_ok, f"Ground-truth leakage detected in {name}: {leak_issues}"
        print("  Zero ground-truth XML report leakage verified across all payloads.")

        # STAGE 20: Verify zero secret leakage
        print("\n[STAGE 20/26] Auditing for zero secret / API key / token leakage...")
        for name, payload in [
            ("experiment_record", fin_exp),
            ("snapshot_manifest", snap),
            ("model_record", m1),
            ("dataset_version_record", fin_dsv)
        ]:
            sec_ok, sec_issues = Phase17Validator.verify_zero_secrets(payload)
            assert sec_ok, f"Secret leakage detected in {name}: {sec_issues}"
        print("  Zero secret / private key leakage verified across all artifacts.")

        # STAGE 21: Verify machine artifact SHA-256 immutability
        print("\n[STAGE 21/26] Verifying machine evidence artifact byte-level immutability...")
        iu_hashes_after = compute_dir_sha256(iu_xray_dir)
        immut_ok, immut_issues = Phase17Validator.verify_machine_artifacts_immutability(iu_hashes_before, iu_hashes_after)
        assert immut_ok, f"Machine artifact byte-level mutation detected: {immut_issues}"
        print(f"  Byte-level immutability verified for all {len(iu_hashes_before)} machine evidence files.")

        # STAGE 22: Verify snapshot cryptographic fingerprint
        print("\n[STAGE 22/26] Verifying snapshot integrity & SHA-256 digest...")
        snap_ver = snap_mgr.verify_snapshot("exp_e2e_phase17_main")
        assert snap_ver["manifest_valid"] is True
        assert snap_ver["is_valid"] is True
        print(f"  Snapshot verification: Valid={snap_ver['is_valid']}, SHA-256={snap_ver['manifest_sha256'][:16]}...")

        # STAGE 23: Test Phase 1.7 REST API Endpoints
        print("\n[STAGE 23/26] Testing live Phase 1.7 REST API endpoints...")
        port = 8945
        server = create_server(port=port)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        time.sleep(0.3)

        base_url = f"http://127.0.0.1:{port}"
        try:
            # Test GET /api/experiment-dashboard
            with urlopen(f"{base_url}/api/experiment-dashboard", timeout=5) as resp:
                assert resp.status == 200
                dash = json.loads(resp.read().decode("utf-8"))
                assert "total_experiments" in dash
                assert "total_models" in dash
                assert "total_dataset_versions" in dash

            # Test GET /api/models
            with urlopen(f"{base_url}/api/models", timeout=5) as resp:
                assert resp.status == 200
                models_list = json.loads(resp.read().decode("utf-8"))
                assert "models" in models_list

            # Test GET /api/dataset-versions
            with urlopen(f"{base_url}/api/dataset-versions", timeout=5) as resp:
                assert resp.status == 200
                dsv_list = json.loads(resp.read().decode("utf-8"))
                assert "dataset_versions" in dsv_list

            # Test GET /api/experiments
            with urlopen(f"{base_url}/api/experiments", timeout=5) as resp:
                assert resp.status == 200
                exps_list = json.loads(resp.read().decode("utf-8"))
                assert "experiments" in exps_list

            print("  All Phase 1.7 REST API endpoints returned HTTP 200 with schema compliance.")
        finally:
            server.shutdown()
            server.server_close()

        # STAGE 24: Test JSON export
        print("\n[STAGE 24/26] Testing structured JSON experiment export package...")
        json_str, mime = exp_runner.generate_export_package("exp_e2e_phase17_main", format_type="json")
        assert mime == "application/json"
        parsed_exp = json.loads(json_str)
        assert parsed_exp["experiment_id"] == "exp_e2e_phase17_main"
        print("  Structured JSON export verified.")

        # STAGE 25: Test plain-text report export
        print("\n[STAGE 25/26] Testing plain-text research report export...")
        txt_str, mime_txt = exp_runner.generate_export_package("exp_e2e_phase17_main", format_type="text")
        assert mime_txt == "text/plain"
        assert "RESEARCH EXPERIMENT RECORD" in txt_str
        assert "MANDATORY RESEARCH DISCLAIMER" in txt_str
        print("  Plain-text research report export verified with required disclaimers.")

        # STAGE 26: Verify reproducibility by re-generating fingerprint
        print("\n[STAGE 26/26] Verifying full reproducibility from stored metadata...")
        reproduced_fp = exp_reg.generate_experiment_fingerprint(fin_exp)
        assert reproduced_fp == fin_exp["fingerprint_sha256"], "Reproduced fingerprint must match exactly"
        print(f"  Fingerprint exact match confirmed: {reproduced_fp}")

        print("\n" + "=" * 80)
        print("ALL 26/26 STAGES COMPLETED AND VERIFIED SUCCESSFULLY!")
        print("PHASE 1.7 EXPERIMENT REGISTRY & VERSIONING: 100% OPERATIONAL")
        print("=" * 80)
        return True

    finally:
        shutil.rmtree(test_temp_dir, ignore_errors=True)


if __name__ == "__main__":
    success = run_e2e_pipeline()
    sys.exit(0 if success else 1)
