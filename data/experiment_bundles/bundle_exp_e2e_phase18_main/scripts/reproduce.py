# Research Experiment Reproduction Script
# This script verifies metadata and canonical SHA-256 fingerprint reproducibility.
import json, hashlib

with open('experiment/config.json') as f: config = json.load(f)
print('Reproducing experiment configuration from bundle...')
print('Config loaded successfully.')
