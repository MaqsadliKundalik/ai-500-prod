"""
Train Price Anomaly Detection Models
=====================================
Two approaches:
1. Isolation Forest - Fast, unsupervised, good for outliers
2. Autoencoder - Deep learning, learns normal price patterns
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Dict, Tuple


class PriceDataset(Dataset):
    """PyTorch dataset for price data."""
    
    def __init__(self, features):
        self.features = torch.FloatTensor(features)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]


class PriceAutoencoder(nn.Module):
    """Autoencoder for price anomaly detection."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32, 16]):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        for hidden_dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def load_and_prepare_data(data_file: str = "datasets/pharmacy_prices_synthetic.json") -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """Load and prepare price data for training."""
    
    print("Loading dataset...")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} price records")
    
    # Feature engineering
    print("Engineering features...")
    
    # Encode categorical variables
    encoders = {}
    for col in ['region', 'pharmacy', 'inn', 'atx_code']:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # Create additional features
    df['price_per_base'] = df['price'] / df['base_price']
    df['log_price'] = np.log1p(df['price'])
    df['log_base_price'] = np.log1p(df['base_price'])
    df['price_deviation'] = df['price'] - df['base_price']
    df['price_deviation_log'] = np.log1p(np.abs(df['price_deviation']))
    
    # Select features for models
    feature_columns = [
        'region_encoded', 'pharmacy_encoded', 'inn_encoded', 'atx_code_encoded',
        'base_price', 'price', 'log_price', 'log_base_price',
        'price_per_base', 'price_deviation', 'price_deviation_log'
    ]
    
    features = df[feature_columns].values
    
    print(f"Features shape: {features.shape}")
    print(f"Anomalies in dataset: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.2f}%)")
    
    return df, features, encoders


def train_isolation_forest(features: np.ndarray, df: pd.DataFrame) -> IsolationForest:
    """Train Isolation Forest model."""
    
    print("\n" + "="*60)
    print("Training Isolation Forest")
    print("="*60)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,  # Expected proportion of anomalies
        max_samples='auto',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("Training model...")
    model.fit(features_scaled)
    
    # Predict
    predictions = model.predict(features_scaled)
    anomaly_scores = model.score_samples(features_scaled)
    
    # Convert predictions (-1 for anomaly, 1 for normal)
    predicted_anomalies = (predictions == -1)
    
    # Evaluation
    true_anomalies = df['is_anomaly'].values
    
    tp = np.sum((predicted_anomalies) & (true_anomalies))
    fp = np.sum((predicted_anomalies) & (~true_anomalies))
    tn = np.sum((~predicted_anomalies) & (~true_anomalies))
    fn = np.sum((~predicted_anomalies) & (true_anomalies))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nIsolation Forest Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    
    # Save model and scaler
    joblib.dump(model, 'models/isolation_forest_price_anomaly.pkl')
    joblib.dump(scaler, 'models/price_scaler.pkl')
    print("\nSaved: models/isolation_forest_price_anomaly.pkl")
    
    return model


def train_autoencoder(features: np.ndarray, df: pd.DataFrame, epochs: int = 50) -> PriceAutoencoder:
    """Train Autoencoder model."""
    
    print("\n" + "="*60)
    print("Training Autoencoder")
    print("="*60)
    
    # Use only normal samples for training
    normal_features = features[~df['is_anomaly'].values]
    print(f"Training on {len(normal_features)} normal samples")
    
    # Standardize
    scaler = StandardScaler()
    normal_features_scaled = scaler.fit_transform(normal_features)
    all_features_scaled = scaler.transform(features)
    
    # Split data
    train_features, val_features = train_test_split(normal_features_scaled, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = PriceDataset(train_features)
    val_dataset = PriceDataset(val_features)
    test_dataset = PriceDataset(all_features_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = PriceAutoencoder(input_dim=features.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/autoencoder_price_anomaly.pt')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Load best model
    model.load_state_dict(torch.load('models/autoencoder_price_anomaly.pt'))
    
    # Calculate reconstruction errors
    model.eval()
    reconstruction_errors = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            errors = torch.mean((output - batch) ** 2, dim=1)
            reconstruction_errors.extend(errors.cpu().numpy())
    
    reconstruction_errors = np.array(reconstruction_errors)
    
    # Set threshold (95th percentile of normal samples)
    normal_errors = reconstruction_errors[~df['is_anomaly'].values]
    threshold = np.percentile(normal_errors, 95)
    
    # Predictions
    predicted_anomalies = reconstruction_errors > threshold
    true_anomalies = df['is_anomaly'].values
    
    # Evaluation
    tp = np.sum((predicted_anomalies) & (true_anomalies))
    fp = np.sum((predicted_anomalies) & (~true_anomalies))
    tn = np.sum((~predicted_anomalies) & (~true_anomalies))
    fn = np.sum((~predicted_anomalies) & (true_anomalies))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nAutoencoder Results:")
    print(f"Threshold: {threshold:.6f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    
    # Save threshold and scaler
    joblib.dump({'threshold': threshold, 'scaler': scaler}, 'models/autoencoder_config.pkl')
    print("\nSaved: models/autoencoder_price_anomaly.pt")
    print("Saved: models/autoencoder_config.pkl")
    
    return model


def main():
    """Main training function."""
    
    # Load data
    df, features, encoders = load_and_prepare_data()
    
    # Save encoders
    joblib.dump(encoders, 'models/price_encoders.pkl')
    print("Saved: models/price_encoders.pkl")
    
    # Train Isolation Forest
    isolation_forest = train_isolation_forest(features, df)
    
    # Train Autoencoder
    autoencoder = train_autoencoder(features, df, epochs=50)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nSaved models:")
    print("  - models/isolation_forest_price_anomaly.pkl")
    print("  - models/autoencoder_price_anomaly.pt")
    print("  - models/price_scaler.pkl")
    print("  - models/autoencoder_config.pkl")
    print("  - models/price_encoders.pkl")


if __name__ == "__main__":
    main()
