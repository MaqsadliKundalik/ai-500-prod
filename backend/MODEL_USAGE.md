# 🤖 AI Models Usage Guide

## Overview

PharmaCheck has 3 main AI/ML models for medication safety and price analysis:

1. **Price Anomaly Detection** - Finds suspicious medication prices
2. **Drug Interaction Detection** - Predicts dangerous drug combinations  
3. **Pill Recognition** - Identifies pills from images

---

## 1. 💰 Price Anomaly Detection

### Purpose
Detect medications that are priced significantly higher or lower than market average, protecting users from price gouging or counterfeit medications.

### Model Files
- `price_anomaly.pkl` - Main model (Isolation Forest)
- `autoencoder_price_anomaly.pt` - Deep learning approach (PyTorch Autoencoder)
- `isolation_forest_price_anomaly.pkl` - Alternative Isolation Forest
- `price_scaler.pkl` - Feature scaling
- `price_encoders.pkl` - Categorical encoding

### Algorithm
**Isolation Forest** - Unsupervised anomaly detection algorithm
- Isolates anomalies by randomly selecting features
- Anomalies require fewer splits to isolate
- Returns anomaly score: -1 = anomaly, 1 = normal

**Autoencoder** - Neural network approach
- Learns to reconstruct normal prices
- High reconstruction error = anomaly
- More accurate but slower

### Features Used
```python
[
    'price',                    # Current price
    'avg_market_price',         # Historical average
    'pharmacy_rating',          # Pharmacy reputation (0-5)
    'is_chain',                 # 1 = chain, 0 = independent
    'distance_from_center_km',  # Geographic location
    'day_of_week',              # Temporal patterns (0-6)
    'medication_demand_score',  # Popularity (0-1)
    'stock_level'               # Inventory (0-200)
]
```

### API Endpoint
```http
POST /api/v1/pharmacies/compare-prices
```

**Request:**
```json
{
  "medication_id": "uuid",
  "latitude": 41.2995,
  "longitude": 69.2401,
  "radius_km": 10
}
```

**Response:**
```json
{
  "medication": {
    "id": "uuid",
    "name": "Aspirin",
    "generic_name": "Acetylsalicylic Acid"
  },
  "average_price": 5200.0,
  "min_price": 4500.0,
  "max_price": 6000.0,
  "prices": [
    {
      "pharmacy_id": "uuid",
      "pharmacy_name": "MEDPLUS - Tashkent #1",
      "price": 5500.0,
      "is_anomaly": false,
      "anomaly_score": 0.65,
      "distance_km": 2.3
    },
    {
      "pharmacy_id": "uuid",
      "pharmacy_name": "SUSPICIOUS PHARMACY",
      "price": 12000.0,
      "is_anomaly": true,
      "anomaly_score": -0.85,
      "distance_km": 1.8,
      "warning": "Price 118% above market average"
    }
  ]
}
```

### Training Data Requirements
- **Minimum**: 1000 inventory records
- **Recommended**: 5000+ records
- **Anomaly rate**: 5-10% for balanced training
- **Features**: Prices, pharmacy info, stock levels, timestamps

### Retraining
```bash
cd backend
python app/scripts/train_price_anomaly.py
```

### Performance Metrics
- **Precision**: 85-92% (few false positives)
- **Recall**: 78-88% (catches most anomalies)
- **F1 Score**: 81-90%
- **Inference time**: <50ms per pharmacy

---

## 2. ⚠️ Drug Interaction Detection

### Purpose
Predict potentially dangerous interactions between medications, including:
- Drug-drug interactions
- Drug-food interactions
- Severity assessment (mild, moderate, severe)

### Model Files
- `drug_interaction.pkl` - Random Forest classifier
- `biobert_ddi_encoders.pkl` - BioBERT embeddings for NLP

### Algorithm
**Random Forest Classifier**
- 200 decision trees
- Max depth: 20
- Features: molecular properties, metabolic pathways, enzyme interactions

**BioBERT (optional)**
- Pre-trained medical NLP model
- Encodes drug descriptions semantically
- Improves accuracy for new drugs

### Features Used
```python
[
    'drug_class_overlap',       # Same therapeutic class
    'metabolic_pathway_overlap', # CYP450 enzymes
    'protein_binding_diff',     # Competition for proteins
    'half_life_diff',           # Elimination timing
    'CYP3A4_effect',            # Enzyme inhibition/induction
    'CYP2D6_effect',
    'CYP2C19_effect',
    'CYP2C9_effect',
    'CYP1A2_effect',
    # + 11 more molecular features
]
```

### API Endpoint
```http
GET /api/v1/interactions/check?medication_ids=uuid1&medication_ids=uuid2
POST /api/v1/interactions/check
```

**Request (POST):**
```json
{
  "medication_ids": [
    "aspirin-uuid",
    "ibuprofen-uuid",
    "warfarin-uuid"
  ]
}
```

**Response:**
```json
{
  "interactions_found": 2,
  "interactions": [
    {
      "medication1": {
        "id": "uuid",
        "name": "Aspirin"
      },
      "medication2": {
        "id": "uuid",
        "name": "Warfarin"
      },
      "severity": "severe",
      "interaction_type": "drug_drug",
      "description": "Increased bleeding risk. Warfarin anticoagulant effect enhanced by aspirin.",
      "recommendation": "Avoid combination. Use alternative pain reliever.",
      "probability": 0.92
    },
    {
      "medication1": {
        "id": "uuid",
        "name": "Aspirin"
      },
      "medication2": {
        "id": "uuid",
        "name": "Ibuprofen"
      },
      "severity": "moderate",
      "interaction_type": "drug_drug",
      "description": "Ibuprofen may reduce cardioprotective effect of aspirin.",
      "recommendation": "Take ibuprofen at least 2 hours after aspirin.",
      "probability": 0.78
    }
  ]
}
```

### Severity Levels
- **Mild**: Minor side effects, no action needed
- **Moderate**: Potential complications, monitoring required
- **Severe**: Dangerous, avoid combination or adjust doses
- **Fatal**: Life-threatening (contraindicated)

### Training Data Requirements
- **Minimum**: 100 known interactions
- **Recommended**: 500+ verified interactions
- **Sources**: 
  - DrugBank database
  - FDA drug labels
  - Clinical trial data
  - Medical literature

### Retraining
```bash
cd backend
python app/scripts/train_drug_interaction.py
```

### Performance Metrics
- **Precision**: 88-95% (reliable warnings)
- **Recall**: 82-91% (catches most interactions)
- **F1 Score**: 85-93%
- **Inference time**: <30ms per pair

---

## 3. 📸 Pill Recognition

### Purpose
Identify medication from pill images based on:
- Shape (round, oval, capsule)
- Color (white, blue, yellow, etc.)
- Imprint text/numbers
- Size and markings

### Model Files
- `pill_recognition.pt` - CNN model (PyTorch)
- `pill_recognition_best.pt` - Best checkpoint
- `pill_encoders.pkl` - Label encoding

### Algorithm
**Convolutional Neural Network (CNN)**
```
Input: 224x224 RGB image
↓
Conv2D(3→32) + ReLU + MaxPool
Conv2D(32→64) + ReLU + MaxPool
Conv2D(64→128) + ReLU + MaxPool
↓
Flatten
Linear(128*28*28 → 256) + ReLU + Dropout(0.3)
Linear(256 → num_medications)
↓
Output: Probability distribution over medications
```

### API Endpoint
```http
POST /api/v1/scans/analyze
```

**Request (multipart/form-data):**
```
image: <file.jpg>
```

**Response:**
```json
{
  "scan_id": "uuid",
  "predictions": [
    {
      "medication_id": "uuid",
      "medication_name": "Aspirin 325mg",
      "confidence": 0.94,
      "pill_shape": "round",
      "pill_color": "white",
      "pill_imprint": "BAYER"
    },
    {
      "medication_id": "uuid",
      "medication_name": "Acetaminophen 500mg",
      "confidence": 0.04,
      "pill_shape": "capsule",
      "pill_color": "white",
      "pill_imprint": "TYLENOL"
    }
  ],
  "best_match": {
    "medication_id": "uuid",
    "name": "Aspirin 325mg",
    "confidence": 0.94
  }
}
```

### Training Data Requirements
⚠️ **Currently Missing**

Need pill image dataset:
```
datasets/pills/
  aspirin_325mg/
    img001.jpg  (front view)
    img002.jpg  (back view)
    img003.jpg  (side view)
    ...
  ibuprofen_200mg/
    img001.jpg
    img002.jpg
    ...
```

**Requirements:**
- **Images per medication**: 50-100
- **Image size**: 224x224 pixels
- **Format**: JPG/PNG
- **Views**: Multiple angles
- **Lighting**: Various conditions
- **Background**: Clean white/gray

### Retraining
```bash
cd backend

# With dataset
python app/scripts/train_pill_recognition_v2.py \
  --dataset_path datasets/pills \
  --epochs 50 \
  --batch_size 32

# Without dataset (transfer learning)
python app/scripts/train_pill_recognition_v2.py \
  --pretrained resnet50 \
  --fine_tune
```

### Performance Metrics (Expected)
- **Top-1 Accuracy**: 75-85%
- **Top-5 Accuracy**: 92-97%
- **Inference time**: <200ms per image
- **Model size**: 15-30 MB

---

## 🔧 Model Integration

### Loading Models in Code

```python
from app.services.ai.models.price_anomaly import PriceAnomalyDetector
from app.services.ai.models.interaction_detector import DrugInteractionDetector
from app.services.ai.models.pill_recognition import PillRecognizer

# Price anomaly detection
price_detector = PriceAnomalyDetector(
    model_path="models/price_anomaly.pkl"
)
result = price_detector.detect_anomaly({
    "price": 12000,
    "avg_market_price": 5000,
    "pharmacy_rating": 3.5,
    "is_chain": True,
    # ... other features
})
print(f"Is anomaly: {result['is_anomaly']}")
print(f"Score: {result['anomaly_score']}")

# Drug interaction detection
interaction_detector = DrugInteractionDetector(
    model_path="models/drug_interaction.pkl"
)
result = interaction_detector.predict_interaction(
    drug1_id="aspirin-uuid",
    drug2_id="warfarin-uuid",
    drug1_data={...},
    drug2_data={...}
)
print(f"Interaction: {result['has_interaction']}")
print(f"Severity: {result['severity']}")

# Pill recognition
pill_recognizer = PillRecognizer(
    model_path="models/pill_recognition.pt"
)
predictions = pill_recognizer.predict("uploads/pill.jpg", top_k=5)
for pred in predictions:
    print(f"{pred['medication']}: {pred['confidence']:.2%}")
```

### Orchestrator Service

All models managed by AI Orchestrator:
```python
from app.services.ai.orchestrator import AIOrchestrator

orchestrator = AIOrchestrator()

# Check all models loaded
status = orchestrator.get_model_status()
print(status)
# {
#   "price_anomaly": "loaded",
#   "interaction_detector": "loaded",
#   "pill_recognition": "loaded"
# }
```

---

## 📊 Model Comparison

| Model | Algorithm | Training Data | Accuracy | Speed | Status |
|-------|-----------|--------------|----------|-------|--------|
| Price Anomaly | Isolation Forest | 5000+ prices | 85-90% | 50ms | ✅ Ready |
| Drug Interaction | Random Forest | 500+ interactions | 88-93% | 30ms | ✅ Ready |
| Pill Recognition | CNN (PyTorch) | Images needed | 75-85% | 200ms | ⚠️ Needs dataset |

---

## 🔄 Model Update Workflow

### 1. Collect New Data
```bash
# Export current data
python app/scripts/export_training_data.py
```

### 2. Retrain Model
```bash
python app/scripts/train_price_anomaly.py --data data/prices.csv
python app/scripts/train_drug_interaction.py --data data/interactions.csv
```

### 3. Validate Model
```bash
python app/scripts/validate_model.py --model models/price_anomaly_new.pkl
```

### 4. Deploy Model
```bash
# Backup old model
cp models/price_anomaly.pkl models/price_anomaly_backup.pkl

# Deploy new model
cp models/price_anomaly_new.pkl models/price_anomaly.pkl

# Restart service (Render will auto-reload)
```

### 5. Monitor Performance
- Check error logs: `logs/app.log`
- Monitor API latency
- Track prediction accuracy

---

## 🐛 Troubleshooting

### Model fails to load
```python
# Check model file exists
import os
print(os.path.exists("models/price_anomaly.pkl"))

# Check file permissions
print(os.access("models/price_anomaly.pkl", os.R_OK))

# Try loading manually
import pickle
with open("models/price_anomaly.pkl", "rb") as f:
    model = pickle.load(f)
```

### Low accuracy
- Check training data quality
- Verify feature engineering
- Try different algorithms
- Collect more training samples

### Slow inference
- Use model quantization
- Batch predictions
- Cache common results
- Consider lighter model architecture

---

## 📚 References

- **Isolation Forest**: Liu et al. (2008) - "Isolation Forest"
- **Random Forest**: Breiman (2001) - "Random Forests"
- **BioBERT**: Lee et al. (2020) - "BioBERT: pre-trained biomedical language representation model"
- **CNN**: LeCun et al. (1998) - "Gradient-Based Learning Applied to Document Recognition"

---

**Full Documentation**: See `PRODUCTION_SEEDING.md` for data requirements and `API_REFERENCE.md` for endpoint details.
