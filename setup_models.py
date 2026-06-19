#!/usr/bin/env python3
"""
setup_models.py — Download and cache the sentence-transformer model.

Run this ONCE before using rank.py. The ranking script loads from disk only
(no network calls at ranking time, per hackathon compute constraints).

Usage:
    python setup_models.py
"""

import os
import sys
from pathlib import Path


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = Path("models/all-MiniLM-L6-v2")


def download_model():
    """Download the sentence-transformer model to local models/ directory."""
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        print(f"✓ Model already exists at {MODEL_DIR}")
        print("  To re-download, delete the directory and run again.")
        return

    print(f"Downloading {MODEL_NAME} to {MODEL_DIR}...")
    print("This is a one-time setup step (~80MB download).")
    print()

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save(str(MODEL_DIR))
        print(f"\n✓ Model saved to {MODEL_DIR}")
        print("  You can now run: python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv")

    except ImportError:
        print("ERROR: sentence-transformers is not installed.")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to download model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_model()
