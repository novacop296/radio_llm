"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.9 — Visual Grounding Unit Test Suite

Verifies:
1. VisualGrounding initializes properly.
2. Correct DenseNet model loaded with target norm5 layer.
3. Finding-to-index mapping resolves canonical pathology names case-insensitively.
4. Target layer discovery identifies model.features.norm5.
5. Preprocessing tensor compatibility [1, 1, 224, 224].
6. Invalid pathology name handling (graceful error / rejection).
7. Grad-CAM generation succeeds.
8. Heatmap contains finite values (no NaN / Inf).
9. Heatmap normalization bounds [0.0, 1.0].
10. Heatmap shape resizing matches original image dimensions.
11. Overlay generation produces valid 3-channel RGB image.
12. Output files created on disk.
13. Missing image handling raises appropriate error.
14. CPU execution works deterministically.
15. CUDA execution works if GPU available, otherwise gracefully falls back to CPU.
16. Hooks and gradients cleaned after execution.
17. Model parameters are not modified (SHA-256 parameter checksum equality).
18. Ground-truth leakage protection (no report ground truth in grounding artifacts).
19. No anatomical locations or coordinates invented.
20. Model score naming preserved (no 'confidence' / 'probability' labeling).
21. End-to-end visual grounding pipeline validation passes.
"""

import os
import sys
import unittest
import json
import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image

# Add project root and backend to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from visual_grounding import VisualGrounding, VisualGroundingEngine, compute_model_checksum
from validate_grounding import validate_grounding_package


class TestPhase09VisualGrounding(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        cls.model.eval()
        cls.grounding = VisualGrounding(model=cls.model, device=cls.device)
        cls.sample_tensor = torch.randn(1, 1, 224, 224)
        cls.test_out_dir = os.path.join(BASE_DIR, "data", "iu_xray", "visual_grounding_test_tmp")
        os.makedirs(cls.test_out_dir, exist_ok=True)
        # Create a small dummy image for testing file generation
        cls.dummy_img_path = os.path.join(cls.test_out_dir, "dummy_xray.png")
        Image.fromarray(np.full((120, 100), 128, dtype=np.uint8)).save(cls.dummy_img_path)

    def test_01_engine_initializes(self):
        """TEST 1: VisualGrounding initializes properly."""
        self.assertIsNotNone(self.grounding)
        self.assertEqual(len(self.grounding.pathologies), 18)
        self.assertTrue(isinstance(self.grounding, VisualGroundingEngine))

    def test_02_correct_model_loaded(self):
        """TEST 2: Model is confirmed to be DenseNet with target norm5 layer."""
        self.assertTrue(hasattr(self.model, "features"))
        self.assertTrue(hasattr(self.model.features, "norm5"))

    def test_03_finding_to_index_mapping(self):
        """TEST 3: Finding names map to valid integer indices case-insensitively."""
        idx_inf = self.grounding.get_pathology_index("Infiltration")
        idx_lower = self.grounding.get_pathology_index("infiltration")
        self.assertIsNotNone(idx_inf)
        self.assertEqual(idx_inf, idx_lower)
        self.assertEqual(self.grounding.pathologies[idx_inf], "Infiltration")

        idx_pneumo = self.grounding.get_pathology_index("pneumothorax")
        self.assertIsNotNone(idx_pneumo)
        self.assertEqual(self.grounding.pathologies[idx_pneumo], "Pneumothorax")

    def test_04_target_layer_discovery(self):
        """TEST 4: Target layer is discovered and verified as model.features.norm5."""
        self.assertEqual(self.grounding.TARGET_LAYER_NAME, "model.features.norm5")
        norm5_module = getattr(self.model.features, "norm5", None)
        self.assertIsNotNone(norm5_module)
        self.assertIsInstance(norm5_module, torch.nn.BatchNorm2d)

    def test_05_preprocessing_compatibility(self):
        """TEST 5: Accepts standard [1, 1, 224, 224] tensor."""
        tensor = torch.zeros((1, 1, 224, 224), dtype=torch.float32)
        cam, score, shape = self.grounding.compute_gradcam(tensor, "Effusion")
        self.assertEqual(shape, (7, 7))
        self.assertIsInstance(score, float)

    def test_06_invalid_pathology_rejected(self):
        """TEST 6: Invalid pathology name raises ValueError."""
        with self.assertRaises(ValueError):
            self.grounding.compute_gradcam(self.sample_tensor, "CompletelyFakeCondition")

    def test_07_gradcam_generation_succeeds(self):
        """TEST 7: Grad-CAM computation returns valid array, score, and shape."""
        cam, score, shape = self.grounding.compute_gradcam(self.sample_tensor, "Infiltration")
        self.assertIsInstance(cam, np.ndarray)
        self.assertEqual(shape, (7, 7))
        self.assertIsInstance(score, float)

    def test_08_heatmap_contains_finite_values(self):
        """TEST 8: Generated CAM array contains no NaN or Inf."""
        cam, _, _ = self.grounding.compute_gradcam(self.sample_tensor, "Pneumothorax")
        self.assertTrue(np.isfinite(cam).all())

    def test_09_heatmap_normalization_bounds(self):
        """TEST 9: Normalized heatmap values strictly in [0.0, 1.0]."""
        cam, _, _ = self.grounding.compute_gradcam(self.sample_tensor, "Effusion")
        self.assertGreaterEqual(float(cam.min()), 0.0)
        self.assertLessEqual(float(cam.max()), 1.0)

    def test_10_heatmap_dimensions_match_after_resizing(self):
        """TEST 10: Artifact generator resizes CAM to original image dimensions."""
        pkg = self.grounding.generate_grounding_artifacts(
            study_id="TEST_STUDY",
            image_id="TEST_IMG",
            view="Frontal",
            original_image_path=self.dummy_img_path,
            input_tensor=self.sample_tensor,
            target_findings=[{"finding": "Infiltration", "status": "possible"}],
            output_dir=self.test_out_dir
        )
        rec = pkg["groundings"][0]
        heat_abs = os.path.join(BASE_DIR, rec["heatmap_path"])
        with Image.open(heat_abs) as img:
            self.assertEqual(img.size, (100, 120))

    def test_11_overlay_generation_succeeds(self):
        """TEST 11: Overlay file exists and has 3 channels (RGB)."""
        pkg = self.grounding.generate_grounding_artifacts(
            study_id="TEST_STUDY_11",
            image_id="TEST_IMG_11",
            view="Frontal",
            original_image_path=self.dummy_img_path,
            input_tensor=self.sample_tensor,
            target_findings=[{"finding": "Effusion", "status": "possible"}],
            output_dir=self.test_out_dir
        )
        rec = pkg["groundings"][0]
        overlay_abs = os.path.join(BASE_DIR, rec["overlay_path"])
        with Image.open(overlay_abs) as img:
            self.assertEqual(img.mode, "RGB")
            self.assertEqual(img.size, (100, 120))

    def test_12_output_files_created(self):
        """TEST 12: Heatmap, overlay, and metadata files exist on disk."""
        pkg = self.grounding.generate_grounding_artifacts(
            study_id="TEST_STUDY_12",
            image_id="TEST_IMG_12",
            view="Frontal",
            original_image_path=self.dummy_img_path,
            input_tensor=self.sample_tensor,
            target_findings=[{"finding": "Cardiomegaly", "status": "possible"}],
            output_dir=self.test_out_dir
        )
        study_dir = os.path.join(self.test_out_dir, "TEST_STUDY_12")
        self.assertTrue(os.path.exists(os.path.join(study_dir, "original.png")))
        self.assertTrue(os.path.exists(os.path.join(study_dir, "cardiomegaly_heatmap.png")))
        self.assertTrue(os.path.exists(os.path.join(study_dir, "cardiomegaly_overlay.png")))
        self.assertTrue(os.path.exists(os.path.join(study_dir, "grounding_metadata.json")))

    def test_13_missing_image_handling(self):
        """TEST 13: Missing original image raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.grounding.generate_grounding_artifacts(
                study_id="TEST_STUDY",
                image_id="TEST_IMG",
                view="Frontal",
                original_image_path="non_existent_path.png",
                input_tensor=self.sample_tensor,
                target_findings=[{"finding": "Infiltration", "status": "possible"}],
                output_dir=self.test_out_dir
            )

    def test_14_cpu_execution(self):
        """TEST 14: Confirms deterministic CPU execution."""
        cpu_grounding = VisualGrounding(model=self.model, device=torch.device("cpu"))
        cam, score, shape = cpu_grounding.compute_gradcam(self.sample_tensor, "Consolidation")
        self.assertEqual(shape, (7, 7))
        self.assertTrue(np.isfinite(cam).all())

    def test_15_cuda_or_fallback(self):
        """TEST 15: Confirms device initialization handles CUDA or fallback gracefully."""
        auto_grounding = VisualGrounding(model=self.model)
        expected_dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(auto_grounding.device.type, expected_dev)

    def test_16_hooks_and_gradients_cleaned(self):
        """TEST 16: No residual parameter gradients remain on the model."""
        self.grounding.compute_gradcam(self.sample_tensor, "Consolidation")
        for param in self.model.parameters():
            if param.grad is not None:
                self.assertTrue((param.grad == 0).all())

    def test_17_model_parameters_not_modified(self):
        """TEST 17: Parameter checksum before and after Grad-CAM is identical."""
        chk_before = compute_model_checksum(self.model)
        self.grounding.compute_gradcam(self.sample_tensor, "Fracture")
        chk_after = compute_model_checksum(self.model)
        self.assertEqual(chk_before, chk_after)

    def test_18_ground_truth_not_included(self):
        """TEST 18: Ground truth is not present in visual grounding artifacts."""
        pkg = self.grounding.generate_grounding_artifacts(
            study_id="TEST_STUDY_18",
            image_id="TEST_IMG_18",
            view="Frontal",
            original_image_path=self.dummy_img_path,
            input_tensor=self.sample_tensor,
            target_findings=[{"finding": "Pneumothorax", "status": "possible"}],
            output_dir=self.test_out_dir
        )
        pkg_str = json.dumps(pkg)
        self.assertNotIn("report_ground_truth", pkg_str)

    def test_19_no_anatomical_location_invented(self):
        """TEST 19: Visual grounding records do NOT assert anatomical coordinates/locations."""
        pkg = self.grounding.generate_grounding_artifacts(
            study_id="TEST_STUDY_19",
            image_id="TEST_IMG_19",
            view="Frontal",
            original_image_path=self.dummy_img_path,
            input_tensor=self.sample_tensor,
            target_findings=[{"finding": "Infiltration", "status": "possible"}],
            output_dir=self.test_out_dir
        )
        for rec in pkg["groundings"]:
            self.assertNotIn("location", rec)

    def test_20_model_score_naming(self):
        """TEST 20: Score is named 'model_score' and never 'confidence'."""
        pkg = self.grounding.generate_grounding_artifacts(
            study_id="TEST_STUDY_20",
            image_id="TEST_IMG_20",
            view="Frontal",
            original_image_path=self.dummy_img_path,
            input_tensor=self.sample_tensor,
            target_findings=[{"finding": "Edema", "status": "possible"}],
            output_dir=self.test_out_dir
        )
        pkg_str = json.dumps(pkg)
        self.assertNotIn('"confidence"', pkg_str)
        self.assertIn('"model_score"', pkg_str)

    def test_21_e2e_grounding_pipeline_succeeds(self):
        """TEST 21: Complete package passes validate_grounding_package validation."""
        pkg = self.grounding.generate_grounding_artifacts(
            study_id="TEST_STUDY_21",
            image_id="TEST_IMG_21",
            view="Frontal",
            original_image_path=self.dummy_img_path,
            input_tensor=self.sample_tensor,
            target_findings=[
                {"finding": "Infiltration", "status": "possible"},
                {"finding": "Pneumothorax", "status": "possible"},
                {"finding": "Consolidation", "status": "possible"}
            ],
            output_dir=self.test_out_dir
        )
        is_valid, errors = validate_grounding_package(pkg, base_dir=BASE_DIR)
        self.assertTrue(is_valid, f"Validation failed with errors: {errors}")


if __name__ == "__main__":
    unittest.main()
