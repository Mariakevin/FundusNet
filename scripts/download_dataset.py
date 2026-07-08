"""
Download and organize the ODIR-5K retinal fundus dataset.

Sources (Kaggle):
  - https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k

Expected output structure under retina_dataset/:
  retina_dataset/
    ├── 1_normal/
    ├── 2_cataract/
    ├── 3_glaucoma/
    └── 4_Diabetic_Retinopathy/
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


DATASET_DIR = Path(__file__).resolve().parent.parent / "retina_dataset"
SUBDIRS = ["1_normal", "2_cataract", "3_glaucoma", "4_Diabetic_Retinopathy"]

# ---------------------------------------------------------------------------
# Mapping from ODIR-5K labels to our directory names
# ODIR-5K uses: N (normal), C (cataract), G (glaucoma), D (diabetic retinopathy)
# ---------------------------------------------------------------------------
LABEL_MAP = {
    "N": "1_normal",
    "C": "2_cataract",
    "G": "3_glaucoma",
    "D": "4_Diabetic_Retinopathy",
}

KAGGLE_URL = "https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k"

MANUAL_INSTRUCTIONS = f"""
{"=" * 70}
  MANUAL DOWNLOAD REQUIRED
{"=" * 70}

  Automatic download via kagglehub failed or is unavailable.

  To download the dataset manually:

  1. Visit: {KAGGLE_URL}
  2. Click "Download" (requires Kaggle account)
  3. Unzip the downloaded file
  4. Organise the images into:

        retina_dataset/1_normal/
        retina_dataset/2_cataract/
        retina_dataset/3_glaucoma/
        retina_dataset/4_Diabetic_Retinopathy/

     Refer to the dataset's labels.csv for the left/right eye label
     columns (N, C, G, D) per image.

  5. Re-run this script -- it will skip existing files.

  Alternatively, install kagglehub and try auto-download:

      pip install kagglehub
      python scripts/download_dataset.py

{"=" * 70}
"""


def dataset_exists():
    """Check whether all expected subdirectories already exist."""
    return all((DATASET_DIR / d).is_dir() for d in SUBDIRS)


def ensure_dirs():
    for d in SUBDIRS:
        (DATASET_DIR / d).mkdir(parents=True, exist_ok=True)


def extract_odir5k_zip(zip_path):
    """Extract and organise ODIR-5K zip into retina_dataset/ subdirectories."""
    import csv

    ensure_dirs()
    zip_path = Path(zip_path)
    extract_root = DATASET_DIR / "_odir_extracted"
    extract_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    # Locate the labels CSV
    csv_candidates = list(extract_root.rglob("*.csv"))
    if not csv_candidates:
        print("[ERROR] No CSV found inside the zip. Cannot map labels.")
        shutil.rmtree(extract_root, ignore_errors=True)
        sys.exit(1)

    csv_path = csv_candidates[0]
    print(f"[INFO] Using labels file: {csv_path}")

    # Read labels -- expected columns: filename, ..., N, C, G, D (one-hot style)
    copied = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in tqdm(list(reader), desc="Organising images"):
            # Determine the label -- ODIR-5K has one-hot indicator per disease
            label = None
            for col, subdir in LABEL_MAP.items():
                val = row.get(col, "0").strip()
                if val == "1":
                    label = subdir
                    break

            if label is None:
                continue  # skip unlabeled or multi-label (rare)

            # Find the corresponding image file
            img_filename = row.get("filename", "").strip()
            if not img_filename:
                continue

            # Search for the image in extracted folders
            found = None
            # Typically ODIR-5K has left/right subfolders inside the zip
            for img_path in extract_root.rglob(img_filename):
                if img_path.is_file():
                    found = img_path
                    break

            if found is None:
                continue

            dest = DATASET_DIR / label / img_filename
            if dest.exists():
                continue

            shutil.copy2(found, dest)
            copied += 1

    # Cleanup extracted zip contents
    shutil.rmtree(extract_root, ignore_errors=True)

    print(f"[INFO] Organised {copied} images into {DATASET_DIR}")
    return copied


def download_via_kagglehub():
    """Try downloading the dataset using kagglehub."""
    import kagglehub

    print("[INFO] Downloading ODIR-5K via kagglehub ...")
    path = kagglehub.dataset_download("andrewmvd/ocular-disease-recognition-odir5k")
    zip_path = Path(path) / "archive.zip"
    if not zip_path.exists():
        # kagglehub may download and extract; check for the dataset folder
        odir_dir = Path(path)
        # If already extracted, re-zip would be wasteful; handle extracted folder
        subdirs = [d for d in odir_dir.iterdir() if d.is_dir()]
        if subdirs:
            print(f"[INFO] kagglehub provided an extracted dataset at {odir_dir}")
            return organise_extracted_folder(odir_dir)
        print("[ERROR] Could not locate the dataset zip or folder.")
        sys.exit(1)

    return extract_odir5k_zip(zip_path)


def organise_extracted_folder(source_dir):
    """Organise a folder that already has images + a CSV into subdirs."""
    import csv

    ensure_dirs()

    csv_candidates = list(Path(source_dir).rglob("*.csv"))
    if not csv_candidates:
        print("[ERROR] No CSV found in downloaded dataset folder.")
        sys.exit(1)

    csv_path = csv_candidates[0]
    copied = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in tqdm(list(reader), desc="Organising images"):
            label = None
            for col, subdir in LABEL_MAP.items():
                val = row.get(col, "0").strip()
                if val == "1":
                    label = subdir
                    break
            if label is None:
                continue

            img_filename = row.get("filename", "").strip()
            if not img_filename:
                continue

            found = None
            for img_path in Path(source_dir).rglob(img_filename):
                if img_path.is_file():
                    found = img_path
                    break

            if found is None:
                continue

            dest = DATASET_DIR / label / img_filename
            if dest.exists():
                continue

            shutil.copy2(found, dest)
            copied += 1

    print(f"[INFO] Organised {copied} images into {DATASET_DIR}")
    return copied


def main():
    parser = argparse.ArgumentParser(description="Download and organise ODIR-5K retinal fundus dataset.")
    parser.add_argument(
        "--zip",
        help="Path to a pre-downloaded ODIR-5K archive.zip (skip download)",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # 1. Check if the dataset already exists
    # ---------------------------------------------------------------
    if dataset_exists():
        print(f"[OK] Dataset already exists at {DATASET_DIR}")
        print("      Subdirectories found:")
        for d in SUBDIRS:
            count = len(list((DATASET_DIR / d).iterdir()))
            print(f"        {d}/  ({count} files)")
        return

    # ---------------------------------------------------------------
    # 2. Zip provided manually?
    # ---------------------------------------------------------------
    if args.zip:
        zip_path = Path(args.zip)
        if not zip_path.is_file():
            print(f"[ERROR] File not found: {zip_path}")
            sys.exit(1)
        extract_odir5k_zip(zip_path)
        return

    # ---------------------------------------------------------------
    # 3. Try automatic download via kagglehub
    # ---------------------------------------------------------------
    try:
        download_via_kagglehub()
        return
    except ImportError:
        pass
    except Exception as exc:
        print(f"[WARN] kagglehub download failed: {exc}")

    # ---------------------------------------------------------------
    # 4. Fallback: print manual instructions
    # ---------------------------------------------------------------
    print(MANUAL_INSTRUCTIONS)


if __name__ == "__main__":
    main()
