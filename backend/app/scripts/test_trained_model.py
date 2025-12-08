"""
Test Trained Pill Recognition Model
====================================
Test the trained model with sample images
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.services.ai.production_pill_recognizer import ProductionPillRecognizer


def main():
    print("🧪 Testing Trained Pill Recognition Model")
    print("=" * 60)
    
    MODEL_PATH = "models/pill_recognition_best.pt"
    ENCODER_PATH = "models/pill_encoders.pkl"
    TEST_IMAGES_DIR = Path("datasets/sample_pills/images")
    
    # Load model
    print("\n📦 Loading model...")
    recognizer = ProductionPillRecognizer(MODEL_PATH, ENCODER_PATH)
    recognizer.load()
    
    print("\n🎯 Model info:")
    print(f"  - Shapes: {list(recognizer.shape_encoder.classes_)}")
    print(f"  - Colors: {list(recognizer.color_encoder.classes_)}")
    print(f"  - Imprints: {list(recognizer.imprint_encoder.classes_)}")
    
    # Test with sample images
    print("\n🔍 Testing with sample images...\n")
    
    test_images = list(TEST_IMAGES_DIR.glob("*.jpg"))[:5]  # Test first 5
    
    for img_path in test_images:
        print(f"\n📸 Image: {img_path.name}")
        print("-" * 60)
        
        # Get basic prediction
        result = recognizer.predict(image_path=str(img_path))
        
        print(f"  Shape:   {result['shape']['prediction']} "
              f"({result['shape']['confidence']:.1%})")
        print(f"  Color:   {result['color']['prediction']} "
              f"({result['color']['confidence']:.1%})")
        print(f"  Imprint: {result['imprint']['prediction']} "
              f"({result['imprint']['confidence']:.1%})")
        print(f"  Combined: {result['combined_confidence']:.1%}")
        
        # Get top-3 predictions
        top_k = recognizer.get_top_k_predictions(image_path=str(img_path), k=3)
        
        print(f"\n  Top 3 Shapes:")
        for item in top_k['shape']:
            print(f"    - {item['label']}: {item['confidence']:.1%}")
    
    print("\n✅ Testing complete!")


if __name__ == "__main__":
    main()
