"""Install CoSS-DF from a cloned GitHub repository in Google Colab.

Mount Google Drive first, clone the repository into /content/coss-df, then run:
    python software/pipeline_source/scripts/bootstrap_colab.py
"""
from pathlib import Path
import os, subprocess, sys

PROJECT = Path(__file__).resolve().parents[1]
DRIVE_ROOT = Path(os.environ.get("COSSDF_DRIVE_ROOT", "/content/drive/MyDrive/CoSS_DF"))
os.environ.setdefault("COSSDF_DRIVE_ROOT", str(DRIVE_ROOT))
os.environ.setdefault("COSSDF_LOCAL_ROOT", "/content/cossdf_work")

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(PROJECT / "requirements-colab.txt")])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "--no-deps", "-e", str(PROJECT)])
print(f"Installed CoSS-DF from {PROJECT}")
print(f"Persistent Drive root: {DRIVE_ROOT}")
