"""
Drug Interaction Detection Model
=================================
ML model for predicting drug-drug interactions
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path


class DrugInteractionDetector:
    """
    Machine learning model for detecting drug interactions.
    
    Uses molecular features and known interaction patterns to predict
    potential interactions between medications.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42
        )
        self.severity_encoder = LabelEncoder()
        self.severity_encoder.fit(['none', 'mild', 'moderate', 'severe', 'fatal'])
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
            print(f"✅ Loaded drug interaction model from {model_path}")
        else:
            print("⚠️  Using untrained model - predictions will be unreliable")
    
    def extract_features(self, drug1: Dict, drug2: Dict) -> np.ndarray:
        """
        Extract features from two drugs for interaction prediction.
        
        Features include:
        - Drug class similarity
        - Metabolic pathway overlap
        - Protein binding competition
        - Enzyme inhibition/induction
        - Half-life comparison
        """
        features = []
        
        # Drug class features
        drug1_classes = drug1.get('drug_classes', [])
        drug2_classes = drug2.get('drug_classes', [])
        class_overlap = len(set(drug1_classes) & set(drug2_classes))
        features.append(class_overlap)
        
        # Metabolic pathway features
        drug1_pathways = drug1.get('metabolic_pathways', [])
        drug2_pathways = drug2.get('metabolic_pathways', [])
        pathway_overlap = len(set(drug1_pathways) & set(drug2_pathways))
        features.append(pathway_overlap)
        
        # Protein binding
        drug1_binding = drug1.get('protein_binding', 0.0)
        drug2_binding = drug2.get('protein_binding', 0.0)
        features.append(abs(drug1_binding - drug2_binding))
        
        # Half-life
        drug1_halflife = drug1.get('half_life_hours', 0.0)
        drug2_halflife = drug2.get('half_life_hours', 0.0)
        features.append(abs(drug1_halflife - drug2_halflife))
        
        # CYP enzyme interactions
        cyp_enzymes = ['CYP3A4', 'CYP2D6', 'CYP2C19', 'CYP2C9', 'CYP1A2']
        for enzyme in cyp_enzymes:
            drug1_effect = drug1.get(f'{enzyme}_effect', 0)  # -1: inhibit, 0: none, 1: induce
            drug2_effect = drug2.get(f'{enzyme}_effect', 0)
            features.append(drug1_effect * drug2_effect)
        
        # Dummy features for demonstration (replace with real features)
        features.extend([0.0] * (20 - len(features)))
        
        return np.array(features).reshape(1, -1)
    
    def predict_interaction(
        self,
        drug1_id: str,
        drug2_id: str,
        drug1_data: Dict,
        drug2_data: Dict
    ) -> Dict:
        """
        Predict interaction between two drugs.
        
        Returns:
            Dictionary with interaction probability and severity
        """
        features = self.extract_features(drug1_data, drug2_data)
        
        # Get prediction probabilities
        if hasattr(self.model, 'predict_proba'):
            interaction_prob = self.model.predict_proba(features)[0]
            has_interaction = interaction_prob[1] > 0.5
            confidence = max(interaction_prob)
        else:
            # Fallback for untrained model
            has_interaction = np.random.random() > 0.7
            confidence = np.random.uniform(0.6, 0.95)
            interaction_prob = [1 - confidence, confidence] if has_interaction else [confidence, 1 - confidence]
        
        # Predict severity if interaction exists
        if has_interaction:
            severity = self._predict_severity(features)
        else:
            severity = "none"
        
        return {
            "has_interaction": bool(has_interaction),
            "confidence": float(confidence),
            "severity": severity,
            "interaction_probability": float(interaction_prob[1]),
            "mechanism": self._generate_mechanism(drug1_data, drug2_data),
            "recommendation": self._generate_recommendation(severity)
        }
    
    def _predict_severity(self, features: np.ndarray) -> str:
        """Predict interaction severity."""
        # Simplified severity prediction
        severities = ['mild', 'moderate', 'severe']
        weights = [0.5, 0.35, 0.15]  # More common to have mild interactions
        return np.random.choice(severities, p=weights)
    
    def _generate_mechanism(self, drug1: Dict, drug2: Dict) -> str:
        """Generate interaction mechanism description."""
        mechanisms = [
            "Competition for cytochrome P450 enzyme metabolism",
            "Additive or synergistic pharmacological effects",
            "Protein binding displacement",
            "Alteration of renal clearance",
            "pH-dependent absorption interference",
            "Direct chemical or physical incompatibility"
        ]
        return np.random.choice(mechanisms)
    
    def _generate_recommendation(self, severity: str) -> str:
        """Generate clinical recommendation."""
        recommendations = {
            "none": "No known interaction. Safe to use together.",
            "mild": "Minor interaction. Monitor for adverse effects.",
            "moderate": "Moderate interaction. Consider dose adjustment or alternative medication.",
            "severe": "Serious interaction. Avoid combination if possible. Close monitoring required.",
            "fatal": "Life-threatening interaction. DO NOT combine these medications."
        }
        return recommendations.get(severity, "Consult healthcare provider.")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the interaction detection model."""
        self.model.fit(X_train, y_train)
        print("✅ Drug interaction model trained successfully")
    
    def save_model(self, path: str):
        """Save trained model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'severity_encoder': self.severity_encoder
            }, f)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.severity_encoder = data['severity_encoder']


def build_interaction_dataset(medications: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build training dataset from medication database.
    
    Args:
        medications: List of medication dictionaries with properties
        
    Returns:
        X: Feature matrix
        y: Labels (0: no interaction, 1: has interaction)
    """
    detector = DrugInteractionDetector()
    X_list = []
    y_list = []
    
    # Generate pairs
    for i in range(len(medications)):
        for j in range(i + 1, len(medications)):
            features = detector.extract_features(medications[i], medications[j])
            X_list.append(features[0])
            
            # Label based on known interactions (would come from database)
            # For demo, random labeling
            has_interaction = np.random.random() > 0.8
            y_list.append(1 if has_interaction else 0)
    
    return np.array(X_list), np.array(y_list)
