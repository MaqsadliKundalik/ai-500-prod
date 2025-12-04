"""
Simplified Training Script - Quick model training
"""
import numpy as np
from pathlib import Path
import sys
import pickle

sys.path.append(str(Path(__file__).parent.parent.parent))

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler


def train_interaction_detector():
    """Train drug interaction detector."""
    print("\n" + "=" * 60)
    print("🚀 Training Drug Interaction Detector")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000  # Reduced for speed
    n_features = 20
    
    print(f"📦 Generating {n_samples} samples with {n_features} features...")
    
    # Simulate drug features
    X = np.random.randn(n_samples, n_features)
    
    # Create interaction patterns
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        # Simple rule: interaction if certain features high
        if X[i, 0] > 0.5 and X[i, 5] > 0.3:
            y[i] = 1  # Interaction
    
    # Add noise
    noise_idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y[noise_idx] = 1 - y[noise_idx]
    
    print(f"✅ Dataset created:")
    print(f"   No interaction: {np.sum(y == 0)} samples")
    print(f"   Interaction: {np.sum(y == 1)} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Train model
    print("\n🏋️  Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=50,  # Reduced from 100
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📈 Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Interaction', 'Interaction'])}")
    
    # Save model
    model_path = Path("models/drug_interaction.pkl")
    model_path.parent.mkdir(exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"💾 Model saved to: {model_path}")
    return model


def train_price_anomaly_detector():
    """Train price anomaly detector."""
    print("\n" + "=" * 60)
    print("🚀 Training Price Anomaly Detector")
    print("=" * 60)
    
    # Generate synthetic price data
    np.random.seed(42)
    n_samples = 800  # Reduced for speed
    n_features = 8
    
    print(f"📦 Generating {n_samples} price records with {n_features} features...")
    
    # Normal prices
    X_normal = np.random.randn(int(n_samples * 0.9), n_features)
    
    # Anomalous prices (overpriced)
    X_anomaly = np.random.randn(int(n_samples * 0.1), n_features) + 3
    
    X = np.vstack([X_normal, X_anomaly])
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✅ Dataset created with ~{int(n_samples * 0.1)} anomalies")
    
    # Train model
    print("\n🏋️  Training Isolation Forest...")
    model = IsolationForest(
        contamination=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)
    
    # Predict
    predictions = model.predict(X_scaled)
    n_anomalies = np.sum(predictions == -1)
    
    print(f"\n📈 Results:")
    print(f"   Detected anomalies: {n_anomalies}/{len(X)} ({n_anomalies/len(X)*100:.1f}%)")
    
    # Save model
    model_path = Path("models/price_anomaly.pkl")
    model_path.parent.mkdir(exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"💾 Model saved to: {model_path}")
    return model


def main():
    """Train all models."""
    print("🎯 Quick Model Training Script")
    print("Training lightweight models for demo...\n")
    
    # Train models
    interaction_model = train_interaction_detector()
    price_model = train_price_anomaly_detector()
    
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print("=" * 60)
    print("\n✅ Models saved to:")
    print("   - models/drug_interaction.pkl")
    print("   - models/price_anomaly.pkl")
    print("\n💡 These models are now ready for use in the backend!")
    print("   Restart the backend to load the trained models.")


if __name__ == "__main__":
    main()
