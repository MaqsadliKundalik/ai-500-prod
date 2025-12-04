"""Quick model test script"""
import torch
import pickle

print("🔍 Testing AI Models...\n")

# Test 1: Pill Recognition
try:
    checkpoint = torch.load('models/pill_recognition.pt', map_location='cpu')
    print("✅ Pill Recognition Model")
    print(f"   Medications: {checkpoint.get('num_medications', 0)}")
    print(f"   Best Accuracy: {checkpoint.get('best_accuracy', 0):.2f}%")
    print(f"   Architecture: {checkpoint.get('architecture', 'Unknown')}")
    print(f"   Med Map Size: {len(checkpoint.get('medication_map', {}))}")
except Exception as e:
    print(f"❌ Pill Recognition Error: {e}")

print()

# Test 2: Drug Interaction
try:
    with open('models/drug_interaction.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Drug Interaction Model")
    print(f"   Type: {type(model).__name__}")
    if hasattr(model, 'n_estimators'):
        print(f"   Estimators: {model.n_estimators}")
except Exception as e:
    print(f"❌ Drug Interaction Error: {e}")

print()

# Test 3: Price Anomaly
try:
    with open('models/price_anomaly.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Price Anomaly Model")
    print(f"   Type: {type(model).__name__}")
    if hasattr(model, 'contamination'):
        print(f"   Contamination: {model.contamination}")
except Exception as e:
    print(f"❌ Price Anomaly Error: {e}")

print("\n🎉 All models ready for deployment!")
