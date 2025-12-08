"""
Create Sample Training Dataset
===============================
Creates a small sample dataset for testing training pipeline
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import json
import random

def create_synthetic_pill_dataset(output_dir: str = "datasets/sample_pills", num_samples: int = 100):
    """
    Create synthetic pill images for testing.
    
    Args:
        output_dir: Output directory
        num_samples: Number of samples to generate
    """
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Define pill characteristics
    shapes = ["round", "oval", "capsule", "oblong"]
    colors = ["white", "blue", "red", "yellow", "green", "pink"]
    imprints = ["A10", "B20", "C30", "D40", "E50", "M100", "N200", "P500"]
    
    medications = []
    
    print(f"Creating {num_samples} synthetic pill images...")
    
    for i in range(num_samples):
        # Random characteristics
        shape = random.choice(shapes)
        color = random.choice(colors)
        imprint = random.choice(imprints)
        
        # Create synthetic image
        if shape == "round":
            size = (200, 200)
        elif shape == "oval":
            size = (250, 180)
        elif shape == "capsule":
            size = (120, 250)
        else:  # oblong
            size = (280, 150)
        
        # Create colored image
        color_map = {
            "white": (255, 255, 255),
            "blue": (100, 150, 255),
            "red": (255, 100, 100),
            "yellow": (255, 255, 100),
            "green": (100, 255, 100),
            "pink": (255, 150, 200)
        }
        
        img_array = np.ones((size[1], size[0], 3), dtype=np.uint8)
        img_array[:] = color_map[color]
        
        # Add some noise
        noise = np.random.randint(-20, 20, img_array.shape, dtype=np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save image
        img = Image.fromarray(img_array)
        img_path = images_dir / f"pill_{i:04d}.jpg"
        img.save(img_path)
        
        # Create metadata
        medications.append({
            "id": i,
            "image": str(img_path.relative_to(output_path)),
            "medication_name": f"Medication_{imprint}_{color}",
            "imprint": imprint,
            "shape": shape,
            "color": color,
            "size_mm": random.uniform(5.0, 15.0)
        })
    
    # Save metadata
    metadata = {
        "total_samples": len(medications),
        "num_shapes": len(shapes),
        "num_colors": len(colors),
        "num_imprints": len(imprints),
        "shapes": shapes,
        "colors": colors,
        "imprints": imprints,
        "medications": medications
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Created {num_samples} samples")
    print(f"📁 Saved to: {output_path}")
    print(f"📊 Stats:")
    print(f"  - Shapes: {len(shapes)}")
    print(f"  - Colors: {len(colors)}")
    print(f"  - Imprints: {len(imprints)}")
    
    return str(output_path)


if __name__ == "__main__":
    print("🔬 Creating Sample Pill Dataset for Training")
    print("=" * 60)
    
    # Create 500 samples for training
    dataset_path = create_synthetic_pill_dataset(
        output_dir="datasets/sample_pills",
        num_samples=500
    )
    
    print("\n✅ Dataset ready for training!")
    print(f"\nNext step: python app/scripts/train_pill_model.py")
