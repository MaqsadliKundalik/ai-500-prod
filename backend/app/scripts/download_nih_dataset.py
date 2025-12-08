"""
Download NIH Pill Image Recognition Dataset
============================================
Downloads and prepares the official NIH dataset for training
"""

import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import json
import shutil

def download_file(url: str, destination: Path):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def download_nih_dataset():
    """
    Download NIH Pill Image Recognition Challenge Dataset.
    
    This is the official dataset from National Library of Medicine (NLM/NIH).
    Contains 4,000+ high-quality images of pills from multiple angles.
    """
    print("📥 Downloading NIH Pill Image Recognition Dataset")
    print("=" * 60)
    
    # Dataset info
    DATASET_URL = "https://data.lhncbc.nlm.nih.gov/public/Pills/PillProjectDisc1.zip"
    DATASET_DIR = Path("datasets/nih_pills")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    zip_path = DATASET_DIR / "PillProjectDisc1.zip"
    
    # Check if already downloaded
    if zip_path.exists():
        print(f"✅ Dataset already downloaded: {zip_path}")
        extract = input("Extract anyway? (y/n): ")
        if extract.lower() != 'y':
            return
    else:
        print(f"\n📍 Downloading from: {DATASET_URL}")
        print(f"📁 Saving to: {zip_path}")
        print(f"⚠️  Size: ~4 GB - This will take several minutes...\n")
        
        try:
            download_file(DATASET_URL, zip_path)
            print(f"\n✅ Download complete!")
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            print("\n💡 Alternative: Download manually from:")
            print("   https://data.lhncbc.nlm.nih.gov/public/Pills/")
            print(f"   Then save to: {zip_path.absolute()}")
            return
    
    # Extract
    print(f"\n📦 Extracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATASET_DIR)
        print(f"✅ Extracted to: {DATASET_DIR}")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return
    
    # Analyze structure
    print(f"\n📊 Analyzing dataset structure...")
    
    extracted_dir = DATASET_DIR / "PillProjectDisc1"
    if not extracted_dir.exists():
        # Sometimes extracts directly
        extracted_dir = DATASET_DIR
    
    # Count files
    image_files = list(extracted_dir.rglob("*.jpg")) + list(extracted_dir.rglob("*.png"))
    csv_files = list(extracted_dir.rglob("*.csv"))
    
    print(f"\n📈 Dataset Statistics:")
    print(f"  - Total images: {len(image_files)}")
    print(f"  - CSV files: {len(csv_files)}")
    
    if csv_files:
        print(f"\n📋 CSV files found:")
        for csv in csv_files[:5]:
            print(f"  - {csv.relative_to(DATASET_DIR)}")
    
    # Create metadata summary
    metadata = {
        "dataset_name": "NIH Pill Image Recognition Challenge",
        "source": "National Library of Medicine (NLM)",
        "url": DATASET_URL,
        "total_images": len(image_files),
        "csv_files": [str(f.relative_to(DATASET_DIR)) for f in csv_files],
        "image_extensions": [".jpg", ".png"],
        "downloaded_at": Path(zip_path).stat().st_mtime if zip_path.exists() else None
    }
    
    with open(DATASET_DIR / "dataset_info.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Dataset ready!")
    print(f"📁 Location: {DATASET_DIR.absolute()}")
    print(f"\nNext step: python app/scripts/prepare_nih_data.py")


if __name__ == "__main__":
    print("🔬 NIH Pill Image Recognition Dataset Downloader")
    print("=" * 60)
    print("\n⚠️  WARNING: This will download ~4 GB of data")
    print("Make sure you have:")
    print("  1. Stable internet connection")
    print("  2. At least 10 GB free disk space")
    print("  3. Time (10-30 minutes depending on speed)")
    
    proceed = input("\nProceed with download? (y/n): ")
    
    if proceed.lower() == 'y':
        download_nih_dataset()
    else:
        print("\n❌ Download cancelled")
        print("\n💡 Alternative: Use smaller datasets for testing:")
        print("   - Kaggle Pill Image Dataset")
        print("   - Custom O'zbek medications dataset")
