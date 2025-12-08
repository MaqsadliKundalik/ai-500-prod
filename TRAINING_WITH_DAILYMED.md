# Training Pill Recognition with DailyMed Data

## Overview

Bu guide DailyMed ma'lumotlaridan foydalanib pill recognition modelni train qilish jarayonini tushuntiradi.

## 🎯 Training Pipeline

```
1. Collect Data → 2. Prepare Dataset → 3. Train Model → 4. Deploy
   (DailyMed API)    (Images + Labels)    (PyTorch CNN)    (Production)
```

## Step 1: Data Collection

### Install Dependencies

```bash
cd backend
pip install httpx pillow tqdm
```

### Run Collection Script

```bash
python app/scripts/collect_dailymed_data.py
```

Bu script:
- ✅ DailyMed API'dan 28 ta eng ko'p ishlatiladigan dorilar haqida ma'lumot to'playdi
- ✅ Har bir dori uchun ~20 ta medication variant
- ✅ Dori rasmlarini yuklab oladi
- ✅ Pill features (imprint, shape, color, size) extract qiladi
- ✅ `datasets/dailymed/` papkasiga saqlaydi

### Output Structure

```
datasets/dailymed/
├── metadata.json          # All medication data
└── images/
    ├── aspirin_xxx_0.jpg
    ├── aspirin_xxx_1.jpg
    ├── ibuprofen_yyy_0.jpg
    └── ...
```

### Metadata Format

```json
{
  "collected_at": "2025-12-07T10:30:00",
  "total_medications": 560,
  "medications": [
    {
      "setid": "xxx-yyy-zzz",
      "name": "ASPIRIN",
      "generic_name": "ASPIRIN",
      "brand_name": "Bayer Aspirin",
      "ndc_codes": ["0002-3227-30"],
      "features": {
        "imprint": "APO 500",
        "shape": "round",
        "color": "white",
        "size_mm": 10.0,
        "score": true
      },
      "images": [
        "images/aspirin_xxx_0.jpg",
        "images/aspirin_xxx_1.jpg"
      ]
    }
  ]
}
```

## Step 2: Train Model

### Install Training Dependencies

```bash
pip install torch torchvision scikit-learn
```

### Run Training Script

```bash
python app/scripts/train_with_dailymed.py
```

Bu script:
- ✅ DailyMed datasetni yuklaydi
- ✅ Multi-task CNN modelni train qiladi:
  - Shape classification (round, oval, capsule, etc.)
  - Color classification (white, blue, red, etc.)
  - Imprint classification (most important!)
- ✅ Transfer learning (EfficientNet-B0 backbone)
- ✅ Data augmentation (rotation, flip, color jitter)
- ✅ Best modelni saqlaydi: `models/pill_recognition_dailymed_best.pt`

### Training Configuration

```python
# Model Architecture
Backbone: EfficientNet-B0 (pretrained on ImageNet)
Classifiers: 3 separate heads (shape, color, imprint)

# Loss Weights
Shape: 20%
Color: 20%
Imprint: 60%  # Most important!

# Training
Epochs: 50
Batch Size: 32
Optimizer: AdamW (lr=1e-4)
Scheduler: CosineAnnealingLR
```

### Expected Results

```
📊 Training Statistics:
  Total Medications: 560
  Total Images: 2,800
  Train Samples: 2,240
  Val Samples: 560

🎯 Final Accuracy:
  Shape: ~85%
  Color: ~90%
  Imprint: ~75%
  
⏱️ Training Time: ~2 hours (GPU) / ~8 hours (CPU)
```

## Step 3: Use Trained Model

### Load Model

```python
import torch
from app.scripts.train_with_dailymed import MultiTaskPillRecognizer
import pickle

# Load model
checkpoint = torch.load('models/pill_recognition_dailymed_best.pt')
model = MultiTaskPillRecognizer(
    num_shapes=10,  # From encoders
    num_colors=12,
    num_imprints=500
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load label encoders
with open('models/pill_recognition_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
    shape_encoder = encoders['shape_encoder']
    color_encoder = encoders['color_encoder']
    imprint_encoder = encoders['imprint_encoder']
```

### Make Predictions

```python
from PIL import Image
from torchvision import transforms

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('pill.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    shape_out, color_out, imprint_out = model(image_tensor)
    
    shape_pred = shape_encoder.inverse_transform([shape_out.argmax().item()])[0]
    color_pred = color_encoder.inverse_transform([color_out.argmax().item()])[0]
    imprint_pred = imprint_encoder.inverse_transform([imprint_out.argmax().item()])[0]
    
    print(f"Shape: {shape_pred}")
    print(f"Color: {color_pred}")
    print(f"Imprint: {imprint_pred}")
```

## Step 4: Integrate with Multi-Modal Recognition

Update `pill_recognition_multi_modal.py`:

```python
class MultiModalPillRecognizer:
    def __init__(self, visual_model=None, ...):
        # Load DailyMed-trained model
        if visual_model is None:
            self.visual_model = self._load_dailymed_model()
    
    def _load_dailymed_model(self):
        """Load model trained on DailyMed data."""
        checkpoint = torch.load('models/pill_recognition_dailymed_best.pt')
        model = MultiTaskPillRecognizer(...)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
```

## Advanced: Continuous Training

### Daily Updates

Schedule automated collection and retraining:

```bash
# Cron job (daily at 2 AM)
0 2 * * * cd /app && python app/scripts/collect_dailymed_data.py
0 4 * * * cd /app && python app/scripts/train_with_dailymed.py
```

### Incremental Learning

Add new medications without full retrain:

```python
# Fine-tune on new data
model.load_state_dict(torch.load('models/pill_recognition_dailymed_best.pt')['model_state_dict'])

# Freeze backbone, only train classifiers
for param in model.backbone.parameters():
    param.requires_grad = False

# Train on new data
train_model(model, new_data_loader, epochs=10)
```

## Performance Optimization

### 1. Data Augmentation

Increase training data:
```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(360),  # Pills can be any orientation
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 2. Class Balancing

Handle imbalanced data:
```python
from torch.utils.data import WeightedRandomSampler

# Calculate class weights
class_counts = [...]
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [weights[label] for label in labels]

sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=32)
```

### 3. Mixed Precision Training

Faster training:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Evaluation Metrics

### Calculate Metrics

```python
from sklearn.metrics import classification_report, confusion_matrix

# Collect predictions
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for batch in val_loader:
        outputs = model(batch['image'])
        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['imprint'].cpu().numpy())

# Classification report
print(classification_report(all_labels, all_preds))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
```

### Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot confusion matrix
plt.figure(figsize=(20, 20))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.title('Pill Imprint Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
```

## Troubleshooting

### Issue: Not Enough Training Data

**Solution**: Increase `limit_per_drug` in collection script:
```python
await collector.collect_medications(
    limit_per_drug=100,  # Instead of 20
    download_images=True
)
```

### Issue: Low Imprint Accuracy

**Reasons**:
1. Imprint text too small in images
2. Many imprints look similar
3. Not enough samples per imprint

**Solutions**:
1. Use higher resolution images
2. Combine with OCR (our multi-modal approach!)
3. Use metric learning (triplet loss)

### Issue: Out of Memory

**Solution**: Reduce batch size:
```python
train_loader = DataLoader(train_dataset, batch_size=16)  # Instead of 32
```

Or use gradient accumulation:
```python
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Cost Estimation

### Data Collection
- API Calls: Free (DailyMed)
- Images: ~2,800 images × 50KB = ~140 MB
- Time: ~1 hour

### Training
- GPU (Tesla T4): $0.35/hour × 2 hours = **$0.70**
- CPU (16 cores): Free but slower (8 hours)
- Storage: ~500 MB (model + data)

### Total Cost
- First training: **$0.70**
- Daily updates: **$0.70/day**
- Monthly: **~$21**

Very affordable! 💰

## Next Steps

1. ✅ Collect DailyMed data
2. ✅ Train initial model
3. ⏳ Evaluate on test set
4. ⏳ Integrate with multi-modal system
5. ⏳ Deploy to production
6. ⏳ Monitor performance
7. ⏳ Retrain monthly with new data

## Resources

- **DailyMed API**: https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm
- **EfficientNet Paper**: https://arxiv.org/abs/1905.11946
- **PyTorch Docs**: https://pytorch.org/docs/stable/index.html
- **Transfer Learning Guide**: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
