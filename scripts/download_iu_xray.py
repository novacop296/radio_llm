"""
Download and extract the official Indiana University Chest X-Ray (IU X-Ray / Open-i) dataset
from the National Library of Medicine (NLM / NIH) using streaming extraction and archive caching.

Archives:
- Reports: https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz (~1.1 MB)
- Images:  https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz (~1.27 GB)
"""

import os
import sys
import tarfile
import requests
from tqdm import tqdm


REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
IMAGES_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def download_and_extract_reports(reports_url: str, extract_dir: str):
    """Download and extract XML reports."""
    print(f"Acquiring reports from: {reports_url}")
    os.makedirs(extract_dir, exist_ok=True)

    r = requests.get(reports_url, headers=HEADERS, stream=True, timeout=30)
    r.raise_for_status()

    with tarfile.open(mode="r|gz", fileobj=r.raw) as tar:
        count = 0
        for member in tar:
            if member.name.endswith(".xml"):
                tar.extract(member, path=extract_dir)
                count += 1
    print(f"Successfully extracted {count} XML reports to {extract_dir}.\n")


def stream_extract_images(images_url: str, extract_dir: str):
    """Stream download and extract PNG images on the fly."""
    print(f"Streaming and extracting images from: {images_url}")
    os.makedirs(extract_dir, exist_ok=True)

    r = requests.get(images_url, headers=HEADERS, stream=True, timeout=60)
    r.raise_for_status()

    with tarfile.open(mode="r|gz", fileobj=r.raw) as tar, tqdm(total=7470, desc="Extracting Images", unit="img") as pbar:
        for member in tar:
            if member.name.endswith(".png"):
                tar.extract(member, path=extract_dir)
                pbar.update(1)

    print(f"\nAll images extracted successfully to {extract_dir}.\n")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "iu_xray")
    reports_extract_dir = os.path.join(data_dir, "reports")
    images_extract_dir = os.path.join(data_dir, "images")

    os.makedirs(reports_extract_dir, exist_ok=True)
    os.makedirs(images_extract_dir, exist_ok=True)

    print("=" * 60)
    print("IU X-RAY (OPEN-I) DATASET ACQUISITION")
    print("=" * 60)

    # 1. Reports
    print("\n--- 1/2: ACQUIRING RADIOLOGY REPORTS ---")
    if not os.path.exists(os.path.join(reports_extract_dir, "ecgen-radiology")):
        download_and_extract_reports(REPORTS_URL, reports_extract_dir)
    else:
        print("Reports already extracted. Skipping report extraction.\n")

    # 2. Images
    print("\n--- 2/2: ACQUIRING CHEST X-RAY IMAGES ---")
    existing_pngs = len([f for f in os.listdir(images_extract_dir) if f.endswith(".png")])
    if existing_pngs >= 7000:
        print(f"Found {existing_pngs} images already extracted. Skipping image download.\n")
    else:
        stream_extract_images(IMAGES_URL, images_extract_dir)

    print("=" * 60)
    print("DATASET ACQUISITION COMPLETE")
    print(f"Reports directory: {reports_extract_dir}")
    print(f"Images directory : {images_extract_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
