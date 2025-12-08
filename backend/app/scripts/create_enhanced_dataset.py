"""
Create Enhanced Training Dataset
=================================
Creates more realistic synthetic dataset with better variety
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json
import random
from tqdm import tqdm


def create_realistic_pill_image(
    shape: str,
    color: tuple,
    imprint: str,
    size: tuple,
    save_path: Path
):
    """Create more realistic pill image with imprint."""
    
    # Create base image
    if shape == "round":
        img = Image.new('RGB', (300, 300), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw pill (circle with gradient effect)
        for i in range(10):
            radius = 120 - i * 3
            shade = tuple(max(0, c - i * 5) for c in color)
            draw.ellipse(
                [(150 - radius, 150 - radius), (150 + radius, 150 + radius)],
                fill=shade
            )
        
        # Add imprint
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), imprint, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        draw.text(
            (150 - text_width // 2, 150 - text_height // 2),
            imprint,
            fill=(50, 50, 50),
            font=font
        )
        
    elif shape == "oval":
        img = Image.new('RGB', (350, 250), 'white')
        draw = ImageDraw.Draw(img)
        
        for i in range(10):
            offset = i * 3
            shade = tuple(max(0, c - i * 5) for c in color)
            draw.ellipse(
                [(50 + offset, 50 + offset), (300 - offset, 200 - offset)],
                fill=shade
            )
        
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except:
            font = ImageFont.load_default()
        
        draw.text((130, 100), imprint, fill=(50, 50, 50), font=font)
        
    elif shape == "capsule":
        img = Image.new('RGB', (150, 300), 'white')
        draw = ImageDraw.Draw(img)
        
        # Top half
        for i in range(10):
            offset = i * 2
            shade = tuple(max(0, c - i * 5) for c in color)
            draw.ellipse(
                [(30 + offset, 20 + offset), (120 - offset, 150 - offset)],
                fill=shade
            )
        
        # Bottom half (different color)
        bottom_color = tuple(min(255, c + 30) for c in color)
        for i in range(10):
            offset = i * 2
            shade = tuple(max(0, c - i * 5) for c in bottom_color)
            draw.ellipse(
                [(30 + offset, 150 + offset), (120 - offset, 280 - offset)],
                fill=shade
            )
        
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        draw.text((35, 80), imprint[:3], fill=(50, 50, 50), font=font)
        draw.text((35, 190), imprint[3:], fill=(50, 50, 50), font=font)
        
    else:  # oblong
        img = Image.new('RGB', (320, 180), 'white')
        draw = ImageDraw.Draw(img)
        
        for i in range(10):
            offset = i * 2
            shade = tuple(max(0, c - i * 5) for c in color)
            draw.rounded_rectangle(
                [(40 + offset, 40 + offset), (280 - offset, 140 - offset)],
                radius=30,
                fill=shade
            )
        
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except:
            font = ImageFont.load_default()
        
        draw.text((110, 70), imprint, fill=(50, 50, 50), font=font)
    
    # Add noise
    img_array = np.array(img)
    noise = np.random.randint(-15, 15, img_array.shape, dtype=np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    # Resize to standard size
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    
    # Save
    img.save(save_path, quality=95)


def create_enhanced_dataset(output_dir: str = "datasets/enhanced_pills", num_samples: int = 2000):
    """Create enhanced synthetic dataset with more variety."""
    
    print(f"🎨 Creating Enhanced Pill Dataset ({num_samples} samples)")
    print("=" * 60)
    
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Expanded characteristics
    shapes = ["round", "oval", "capsule", "oblong"]
    
    colors_rgb = {
        "white": (245, 245, 245),
        "blue": (100, 150, 255),
        "red": (255, 100, 100),
        "yellow": (255, 255, 100),
        "green": (100, 255, 100),
        "pink": (255, 150, 200),
        "orange": (255, 165, 80),
        "purple": (180, 100, 255),
        "brown": (165, 115, 80),
        "gray": (150, 150, 150),
    }
    
    # Common real pill imprints (from actual medications)
    imprints = [
        # Numbers
        "10", "20", "25", "50", "100", "200", "500",
        # Letter-Number combos
        "A10", "B20", "C30", "D40", "E50",
        "M30", "M60", "M100",
        "N10", "N20", "N200",
        "P500", "P1000",
        "R10", "S25", "T100",
        # Common pharma codes
        "APO", "TEV", "PAR", "WAT", "DAN",
        "G", "K", "L", "M", "N", "R", "S", "T",
        # Two-letter codes
        "AC", "BP", "CP", "DP",
        # Numbers only
        "44", "93", "54", "123", "271",
    ]
    
    medications = []
    
    print(f"Generating images...")
    for i in tqdm(range(num_samples)):
        # Random characteristics
        shape = random.choice(shapes)
        color_name = random.choice(list(colors_rgb.keys()))
        color_rgb = colors_rgb[color_name]
        imprint = random.choice(imprints)
        
        # Vary color slightly
        color_varied = tuple(
            max(0, min(255, c + random.randint(-20, 20)))
            for c in color_rgb
        )
        
        # Create image
        img_path = images_dir / f"pill_{i:05d}.jpg"
        
        try:
            create_realistic_pill_image(
                shape=shape,
                color=color_varied,
                imprint=imprint,
                size=(224, 224),
                save_path=img_path
            )
        except Exception as e:
            print(f"\n⚠️  Failed to create image {i}: {e}")
            continue
        
        # Metadata
        medications.append({
            "id": i,
            "image": str(img_path.relative_to(output_path)),
            "medication_name": f"Medication_{imprint}_{color_name}",
            "imprint": imprint,
            "shape": shape,
            "color": color_name,
            "size_mm": random.uniform(5.0, 20.0),
            "has_score_line": random.choice([True, False]),
            "is_coated": random.choice([True, False])
        })
    
    # Calculate statistics
    metadata = {
        "dataset_name": "Enhanced Synthetic Pills",
        "version": "2.0",
        "total_samples": len(medications),
        "num_shapes": len(shapes),
        "num_colors": len(colors_rgb),
        "num_imprints": len(imprints),
        "shapes": shapes,
        "colors": list(colors_rgb.keys()),
        "imprints": imprints,
        "medications": medications,
        "improvements": [
            "More realistic pill rendering",
            "Gradient shading effects",
            "Visible imprint text",
            "Higher variety of colors",
            "Expanded imprint codes (40+)",
            "Noise and texture added",
            "2000 samples for better training"
        ]
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Created {len(medications)} samples")
    print(f"📁 Saved to: {output_path.absolute()}")
    print(f"📊 Statistics:")
    print(f"  - Shapes: {len(shapes)}")
    print(f"  - Colors: {len(colors_rgb)}")
    print(f"  - Imprints: {len(imprints)}")
    print(f"\nNext step: python app/scripts/train_pill_model.py --dataset enhanced_pills")
    
    return str(output_path)


if __name__ == "__main__":
    create_enhanced_dataset(
        output_dir="datasets/enhanced_pills",
        num_samples=2000
    )
