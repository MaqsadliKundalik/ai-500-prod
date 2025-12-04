"""
Training Script for Drug Interaction Model
==========================================
Train Random Forest classifier for interaction detection
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.services.ai.models.interaction_detector import DrugInteractionDetector


def generate_synthetic_data(n_samples=5000):
    """
    Generate synthetic drug interaction dataset.
    
    Real data sources:
    - DrugBank interaction database
    - FDA Adverse Event Reporting System
    - Published clinical studies
    """
    print("📦 Generating synthetic interaction data...")
    
    # Feature dimensions
    n_features = 20
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels with class imbalance (most pairs don't interact)
    # 80% no interaction, 20% interaction
    y = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    
    # Make some features more predictive
    # If feature 0 and 1 are both high, increase interaction probability
    interaction_score = X[:, 0] * X[:, 1] + X[:, 4] * X[:, 5]
    strong_interaction_mask = interaction_score > 1.5
    y[strong_interaction_mask] = 1
    
    # If feature 2 and 3 are opposite signs, decrease interaction
    opposite_signs = (X[:, 2] * X[:, 3]) < -1.0
    y[opposite_signs] = 0
    
    return X, y


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    predictions = model.model.predict(X_test)
    
    print("\n📊 Model Evaluation:")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, 
                                target_names=['No Interaction', 'Has Interaction']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📈 Detailed Metrics:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   True Negatives: {tn}")
    print(f"   False Positives: {fp}")
    print(f"   False Negatives: {fn}")
    print(f"   True Positives: {tp}")


def main():
    """Main training function."""
    print("🚀 Starting Drug Interaction Model Training")
    print("=" * 60)
    
    # Generate dataset
    X, y = generate_synthetic_data(n_samples=5000)
    print(f"✅ Dataset shape: {X.shape}")
    print(f"✅ Positive samples: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    print(f"✅ Negative samples: {len(y) - np.sum(y)} ({(1-np.mean(y))*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📊 Train set: {len(X_train)} samples")
    print(f"📊 Test set: {len(X_test)} samples")
    
    # Create and train model
    print("\n🏗️  Building Random Forest model...")
    detector = DrugInteractionDetector()
    
    print("🏋️  Training model...")
    detector.train(X_train, y_train)
    
    # Evaluate
    evaluate_model(detector, X_test, y_test)
    
    # Feature importance
    if hasattr(detector.model, 'feature_importances_'):
        importances = detector.model.feature_importances_
        top_features = np.argsort(importances)[-5:][::-1]
        
        print("\n🔍 Top 5 Important Features:")
        for i, idx in enumerate(top_features, 1):
            print(f"   {i}. Feature {idx}: {importances[idx]:.4f}")
    
    # Save model
    model_path = Path("models/drug_interaction.pkl")
    model_path.parent.mkdir(exist_ok=True)
    detector.save_model(str(model_path))
    
    print("\n" + "=" * 60)
    print("🎉 Training completed!")
    print(f"💾 Model saved to: {model_path}")


if __name__ == "__main__":
    main()
