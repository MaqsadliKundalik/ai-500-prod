"""
Prepare NIH Dataset for Training
=================================
Converts NIH dataset format to our training format
"""

import pandas as pd
from pathlib import Path
import json
import shutil
from PIL import Image
from tqdm import tqdm
import re


def parse_nih_dataset():
    """Parse NIH dataset and create training-ready format."""
    
    print("🔧 Preparing NIH Dataset for Training")
    print("=" * 60)
    
    NIH_DIR = Path("datasets/nih_pills")
    OUTPUT_DIR = Path("datasets/nih_prepared")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    images_dir = OUTPUT_DIR / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Find the extracted directory
    extracted_dirs = [
        NIH_DIR / "PillProjectDisc1",
        NIH_DIR / "C0-C9",
        NIH_DIR
    ]
    
    dataset_dir = None
    for d in extracted_dirs:
        if d.exists() and any(d.iterdir()):
            dataset_dir = d
            break
    
    if not dataset_dir:
        print(f"❌ NIH dataset not found in {NIH_DIR}")
        print("Run: python app/scripts/download_nih_dataset.py first")
        return
    
    print(f"📁 Found dataset at: {dataset_dir}")
    
    # Find reference CSV file (usually has metadata)
    csv_files = list(dataset_dir.rglob("*.csv"))
    
    if not csv_files:
        print("⚠️  No CSV metadata found, using image filenames only")
        return process_without_csv(dataset_dir, images_dir)
    
    print(f"\n📋 Found {len(csv_files)} CSV files:")
    for csv in csv_files[:5]:
        print(f"  - {csv.name}")
    
    # Usually the main CSV is named reference.csv or similar
    main_csv = None
    for csv in csv_files:
        if 'reference' in csv.name.lower() or 'pill' in csv.name.lower():
            main_csv = csv
            break
    
    if not main_csv:
        main_csv = csv_files[0]
    
    print(f"\n📊 Using: {main_csv.name}")
    
    # Read CSV
    try:
        df = pd.read_csv(main_csv)
        print(f"✅ Loaded {len(df)} records")
        print(f"\n📋 Columns: {list(df.columns)}")
        print(f"\n🔍 Sample data:")
        print(df.head())
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return process_without_csv(dataset_dir, images_dir)
    
    # Map columns (NIH uses different naming)
    column_mapping = {
        'imprint': ['imprint', 'IMPRINT', 'imprint_code', 'text'],
        'shape': ['shape', 'SHAPE', 'pill_shape'],
        'color': ['color', 'COLOR', 'pill_color', 'color1'],
        'image': ['image', 'filename', 'file', 'image_name'],
        'ndc': ['ndc', 'NDC', 'ndc_code'],
        'name': ['name', 'drug_name', 'medication_name', 'DRUG_NAME']
    }
    
    # Detect actual columns
    detected_cols = {}
    for key, possible_names in column_mapping.items():
        for col in df.columns:
            if col in possible_names or col.lower() in [n.lower() for n in possible_names]:
                detected_cols[key] = col
                break
    
    print(f"\n🎯 Detected columns: {detected_cols}")
    
    # Process images
    medications = []
    image_count = 0
    
    print(f"\n📸 Processing images...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Get image path
            if 'image' in detected_cols:
                img_filename = row[detected_cols['image']]
            else:
                # Try to find image by index
                continue
            
            # Find actual image file
            img_path = None
            for ext in ['.jpg', '.png', '.JPG', '.PNG']:
                potential_path = dataset_dir / img_filename
                if not potential_path.exists():
                    # Try in subdirectories
                    found = list(dataset_dir.rglob(f"*{Path(img_filename).stem}*{ext}"))
                    if found:
                        img_path = found[0]
                        break
                else:
                    img_path = potential_path
                    break
            
            if not img_path or not img_path.exists():
                continue
            
            # Extract metadata
            imprint = row[detected_cols['imprint']] if 'imprint' in detected_cols else "unknown"
            shape = row[detected_cols['shape']] if 'shape' in detected_cols else "unknown"
            color = row[detected_cols['color']] if 'color' in detected_cols else "unknown"
            name = row[detected_cols['name']] if 'name' in detected_cols else "unknown"
            
            # Clean strings
            imprint = str(imprint).strip() if pd.notna(imprint) else "unknown"
            shape = str(shape).strip().lower() if pd.notna(shape) else "unknown"
            color = str(color).strip().lower() if pd.notna(color) else "unknown"
            name = str(name).strip() if pd.notna(name) else "unknown"
            
            # Copy image to output directory
            output_img_name = f"pill_{image_count:05d}{img_path.suffix}"
            output_img_path = images_dir / output_img_name
            
            shutil.copy2(img_path, output_img_path)
            
            medications.append({
                "id": image_count,
                "image": f"images/{output_img_name}",
                "medication_name": name,
                "imprint": imprint,
                "shape": shape,
                "color": color,
                "ndc": row[detected_cols['ndc']] if 'ndc' in detected_cols else None,
                "source": "NIH_Pill_Recognition_Challenge"
            })
            
            image_count += 1
            
        except Exception as e:
            continue
    
    # Save metadata
    metadata = {
        "dataset_name": "NIH Pill Recognition (Prepared)",
        "total_samples": len(medications),
        "num_shapes": len(set(m['shape'] for m in medications if m['shape'] != 'unknown')),
        "num_colors": len(set(m['color'] for m in medications if m['color'] != 'unknown')),
        "num_imprints": len(set(m['imprint'] for m in medications if m['imprint'] != 'unknown')),
        "shapes": sorted(list(set(m['shape'] for m in medications if m['shape'] != 'unknown'))),
        "colors": sorted(list(set(m['color'] for m in medications if m['color'] != 'unknown'))),
        "medications": medications
    }
    
    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Dataset prepared!")
    print(f"📊 Statistics:")
    print(f"  - Total images: {len(medications)}")
    print(f"  - Unique shapes: {metadata['num_shapes']}")
    print(f"  - Unique colors: {metadata['num_colors']}")
    print(f"  - Unique imprints: {metadata['num_imprints']}")
    print(f"\n📁 Output: {OUTPUT_DIR.absolute()}")
    print(f"\nNext step: python app/scripts/train_pill_model.py --dataset nih_prepared")


def process_without_csv(dataset_dir: Path, images_dir: Path):
    """Process images without CSV metadata."""
    print("\n⚠️  Processing without metadata - using filename parsing")
    
    # Find all images
    image_files = []
    for ext in ['*.jpg', '*.png', '*.JPG', '*.PNG']:
        image_files.extend(list(dataset_dir.rglob(ext)))
    
    print(f"📸 Found {len(image_files)} images")
    
    medications = []
    
    for idx, img_path in enumerate(tqdm(image_files[:1000])):  # Limit to 1000 for testing
        try:
            # Try to parse filename for metadata
            filename = img_path.stem
            
            # Copy image
            output_name = f"pill_{idx:05d}{img_path.suffix}"
            shutil.copy2(img_path, images_dir / output_name)
            
            medications.append({
                "id": idx,
                "image": f"images/{output_name}",
                "medication_name": filename,
                "imprint": "unknown",
                "shape": "unknown",
                "color": "unknown",
                "source": "NIH_Pill_Recognition_Challenge"
            })
            
        except Exception as e:
            continue
    
    # Save metadata
    OUTPUT_DIR = images_dir.parent
    metadata = {
        "dataset_name": "NIH Pill Recognition (No Metadata)",
        "total_samples": len(medications),
        "medications": medications,
        "note": "Processed without CSV metadata - labels unknown"
    }
    
    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Images copied: {len(medications)}")
    print(f"📁 Output: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    parse_nih_dataset()
