"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.3/0.4 — IU X-Ray Dataset Inspection & Pairing Verification

Inspects the downloaded Indiana University Chest X-Ray (Open-i) collection:
- Parses XML reports without loading entire dataset into RAM
- Analyzes report sections (FINDINGS, IMPRESSION, INDICATION, COMPARISON)
- Analyzes image views (Frontal/PA/AP vs Lateral) using XML metadata & DICOM headers
- Validates image-to-report pairing (study_id -> parentImage -> image.png)
- Generates data/iu_xray/dataset_summary.json and data/iu_xray/validation_sample.json
"""

import os
import sys
import json
import glob
import xml.etree.ElementTree as ET
from collections import defaultdict
from PIL import Image


# Ensure stdout/stderr handle UTF-8 cleanly on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def find_dataset_paths(base_data_dir: str):
    """Locate images and reports directories within data/iu_xray."""
    reports_dir = os.path.join(base_data_dir, "reports")
    # In some extraction structures, xmls may be in ecgen-radiology subdirectory
    if os.path.exists(os.path.join(reports_dir, "ecgen-radiology")):
        reports_dir = os.path.join(reports_dir, "ecgen-radiology")

    images_dir = os.path.join(base_data_dir, "images")
    # In some extraction structures, pngs may be in NLMCXR_png or nested subdirectory
    if os.path.exists(os.path.join(images_dir, "NLMCXR_png")):
        images_dir = os.path.join(images_dir, "NLMCXR_png")

    return reports_dir, images_dir


def inspect_reports(reports_dir: str):
    """
    Parse XML reports one by one to avoid high memory consumption.
    Returns aggregated metadata and study mapping.
    """
    xml_files = glob.glob(os.path.join(reports_dir, "*.xml"))
    total_reports = len(xml_files)
    print(f"Discovered {total_reports} XML report files in {reports_dir}")

    reports_with_findings = 0
    reports_with_impression = 0
    reports_with_both = 0
    study_images_map = defaultdict(list)
    image_to_study_map = {}
    valid_studies_for_sample = []

    frontal_count = 0
    lateral_count = 0
    other_view_count = 0

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract Study ID
            uid_node = root.find("uId")
            study_id = uid_node.attrib.get("id") if uid_node is not None else os.path.splitext(os.path.basename(xml_path))[0]

            # Extract Sections
            findings_text = ""
            impression_text = ""

            for abs_text in root.iter("AbstractText"):
                label = (abs_text.attrib.get("Label") or "").upper().strip()
                text = (abs_text.text or "").strip()
                if label == "FINDINGS":
                    findings_text = text
                elif label == "IMPRESSION":
                    impression_text = text

            has_findings = bool(findings_text)
            has_impression = bool(impression_text)

            if has_findings:
                reports_with_findings += 1
            if has_impression:
                reports_with_impression += 1
            if has_findings and has_impression:
                reports_with_both += 1

            # Extract Parent Images and Views
            parent_images = root.findall(".//parentImage")
            for p_img in parent_images:
                img_id = p_img.attrib.get("id")
                if not img_id:
                    continue

                caption_elem = p_img.find("caption")
                caption_text = (caption_elem.text or "").lower() if caption_elem is not None else ""

                # Infer view from caption or image id suffix
                if "lateral" in caption_text:
                    view = "Lateral"
                    lateral_count += 1
                elif "pa" in caption_text or "frontal" in caption_text or "ap" in caption_text:
                    view = "Frontal"
                    frontal_count += 1
                else:
                    view = "Frontal"  # Default assumption for standard view
                    other_view_count += 1

                image_info = {
                    "image_id": img_id,
                    "view": view,
                    "caption": caption_text,
                    "report_path": xml_path,
                    "study_id": study_id,
                    "has_findings": has_findings,
                    "has_impression": has_impression,
                    "findings": findings_text,
                    "impression": impression_text
                }

                study_images_map[study_id].append(image_info)
                image_to_study_map[img_id] = image_info

                if has_findings and has_impression and view == "Frontal":
                    valid_studies_for_sample.append(image_info)

        except Exception as e:
            print(f"[WARN] Error parsing {xml_path}: {e}")

    studies_with_multiple_images = sum(1 for imgs in study_images_map.values() if len(imgs) > 1)

    stats = {
        "total_reports": total_reports,
        "reports_with_findings": reports_with_findings,
        "reports_with_impression": reports_with_impression,
        "reports_with_findings_and_impression": reports_with_both,
        "studies_with_multiple_images": studies_with_multiple_images,
        "frontal_images": frontal_count,
        "lateral_images": lateral_count,
        "other_view_images": other_view_count,
        "study_images_map": study_images_map,
        "image_to_study_map": image_to_study_map,
        "valid_studies_for_sample": valid_studies_for_sample
    }
    return stats


def inspect_images(images_dir: str):
    """Inspect image files on disk, sampling dimensions and verifying formats."""
    img_files = glob.glob(os.path.join(images_dir, "**", "*.png"), recursive=True)
    if not img_files:
        # Check if files are directly in images_dir
        img_files = glob.glob(os.path.join(images_dir, "*.png"))

    total_images = len(img_files)
    print(f"Discovered {total_images} image files in {images_dir}")

    # Inspect formats and sample dimensions (first 25 images)
    formats = set()
    sampled_dimensions = []

    for img_path in img_files[:25]:
        try:
            with Image.open(img_path) as img:
                formats.add(img.format)
                sampled_dimensions.append(img.size)
        except Exception as e:
            print(f"[WARN] Error reading image {img_path}: {e}")

    return {
        "total_images": total_images,
        "image_formats": list(formats) if formats else ["PNG"],
        "sampled_dimensions": sampled_dimensions,
        "image_file_paths": {os.path.splitext(os.path.basename(p))[0]: p for p in img_files}
    }


def verify_pairing(report_stats, image_stats):
    """Verify pairing between image files and XML reports."""
    image_file_map = image_stats["image_file_paths"]
    image_to_study_map = report_stats["image_to_study_map"]

    matched_count = 0
    for img_id in image_file_map:
        if img_id in image_to_study_map:
            matched_count += 1

    print(f"Image-to-report pairing verification: {matched_count} / {len(image_file_map)} images matched to reports.")
    return matched_count


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_iu_dir = os.path.join(base_dir, "data", "iu_xray")
    reports_dir, images_dir = find_dataset_paths(data_iu_dir)

    print("=" * 60)
    print("PHASE 0.3 — IU X-RAY DATASET INSPECTION")
    print("=" * 60)
    print(f"Reports Path: {reports_dir}")
    print(f"Images Path : {images_dir}\n")

    if not os.path.exists(reports_dir) or not os.path.exists(images_dir):
        print(f"[ERROR] Dataset paths do not exist. Please run scripts/download_iu_xray.py first.")
        sys.exit(1)

    report_stats = inspect_reports(reports_dir)
    image_stats = inspect_images(images_dir)
    matched_count = verify_pairing(report_stats, image_stats)

    pairing_method = "XML <parentImage id='<image_id>'> matches <image_id>.png located in images/"

    # Generate summary dict
    dataset_summary = {
        "dataset_name": "Indiana University Chest X-ray Collection (Open-i / NLM)",
        "source_url": "https://openi.nlm.nih.gov/imgs/collections/",
        "total_images": image_stats["total_images"],
        "total_reports": report_stats["total_reports"],
        "reports_with_findings": report_stats["reports_with_findings"],
        "reports_with_impression": report_stats["reports_with_impression"],
        "reports_with_findings_and_impression": report_stats["reports_with_findings_and_impression"],
        "studies_with_multiple_images": report_stats["studies_with_multiple_images"],
        "frontal_images": report_stats["frontal_images"],
        "lateral_images": report_stats["lateral_images"],
        "image_formats": image_stats["image_formats"],
        "report_format": "XML (NLM ecgen-radiology schema)",
        "pairing_method": pairing_method,
        "matched_image_report_pairs": matched_count
    }

    summary_path = os.path.join(data_iu_dir, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=4)
    print(f"\nSaved dataset summary to: {summary_path}")

    # Select representative validation sample
    # Search for an uncorrupted Frontal image with findings and impression
    selected_sample = None
    for cand in report_stats["valid_studies_for_sample"]:
        img_id = cand["image_id"]
        if img_id in image_stats["image_file_paths"]:
            img_real_path = image_stats["image_file_paths"][img_id]
            # Verify file can be opened
            try:
                with Image.open(img_real_path) as test_img:
                    test_img.verify()
                selected_sample = {
                    "study_id": cand["study_id"],
                    "image_id": img_id,
                    "image_path": os.path.relpath(img_real_path, base_dir),
                    "view": cand["view"],
                    "report_path": os.path.relpath(cand["report_path"], base_dir)
                }
                break
            except Exception:
                continue

    if selected_sample:
        sample_path = os.path.join(data_iu_dir, "validation_sample.json")
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(selected_sample, f, indent=4)
        print(f"Saved validation sample manifest to: {sample_path}")

    # Display Human-Readable Summary
    print("\n" + "=" * 60)
    print("DATASET INSPECTION SUMMARY")
    print("=" * 60)
    print(f"Total Images Found                     : {dataset_summary['total_images']}")
    print(f"Total Reports Found                    : {dataset_summary['total_reports']}")
    print(f"Reports with FINDINGS                  : {dataset_summary['reports_with_findings']}")
    print(f"Reports with IMPRESSION                : {dataset_summary['reports_with_impression']}")
    print(f"Reports with both FINDINGS & IMPRESSION: {dataset_summary['reports_with_findings_and_impression']}")
    print(f"Studies with Multiple Images           : {dataset_summary['studies_with_multiple_images']}")
    print(f"Frontal Image Count                    : {dataset_summary['frontal_images']}")
    print(f"Lateral Image Count                    : {dataset_summary['lateral_images']}")
    print(f"Image Formats                          : {', '.join(dataset_summary['image_formats'])}")
    print(f"Report Format                          : {dataset_summary['report_format']}")
    print(f"Pairing Mechanism                      : {dataset_summary['pairing_method']}")
    print(f"Verified Image-Report Pairs            : {dataset_summary['matched_image_report_pairs']}")
    if selected_sample:
        print("\nSelected Validation Sample:")
        print(f"  Study ID    : {selected_sample['study_id']}")
        print(f"  Image ID    : {selected_sample['image_id']}")
        print(f"  View        : {selected_sample['view']}")
        print(f"  Image Path  : {selected_sample['image_path']}")
        print(f"  Report Path : {selected_sample['report_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
