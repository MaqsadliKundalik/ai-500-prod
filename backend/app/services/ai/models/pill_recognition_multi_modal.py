"""
Multi-Modal Pill Recognition System
====================================
Complete safety-first approach with 5-step verification
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from PIL import Image
import numpy as np
from datetime import datetime


class VerificationLevel(Enum):
    """Overall confidence level."""
    CRITICAL = "critical"      # Imprint mismatch - DANGER!
    HIGH = "high"             # 4-5 verifications passed
    MEDIUM = "medium"         # 2-3 verifications passed  
    LOW = "low"              # 1 verification passed
    UNVERIFIED = "unverified" # 0 verifications passed


@dataclass
class PillFeatures:
    """Extracted physical features of pill."""
    shape: str                          # round, oval, capsule, oblong
    color_primary: str                  # white, blue, red, etc.
    color_secondary: Optional[str] = None
    diameter_mm: Optional[float] = None
    length_mm: Optional[float] = None
    thickness_mm: Optional[float] = None
    has_imprint: bool = False
    imprint_text: Optional[str] = None  # e.g., "APO 500", "P 500"
    has_score_line: bool = False
    is_coated: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class VerificationResult:
    """Result from one verification method."""
    method: str
    passed: bool
    confidence: float
    details: Dict[str, Any]
    warning: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PillRecognitionResult:
    """Complete pill recognition result."""
    
    # Main identification
    medication_name: str
    medication_id: str
    generic_name: Optional[str] = None
    dosage: Optional[str] = None
    
    # Verification results
    visual_verification: VerificationResult = None
    imprint_verification: Optional[VerificationResult] = None
    size_verification: Optional[VerificationResult] = None
    database_verification: VerificationResult = None
    user_confirmation: Optional[VerificationResult] = None
    
    # Overall assessment
    overall_confidence: float = 0.0
    verification_level: VerificationLevel = VerificationLevel.UNVERIFIED
    verifications_passed: int = 0
    verifications_total: int = 4
    
    # Safety warnings
    similar_medications: List[Dict] = None
    warnings: List[str] = None
    critical_warning: Optional[str] = None
    requires_pharmacist: bool = False
    
    # Metadata
    timestamp: str = None
    processing_time_ms: float = 0.0
    
    def __post_init__(self):
        if self.similar_medications is None:
            self.similar_medications = []
        if self.warnings is None:
            self.warnings = []
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dict for API response."""
        data = asdict(self)
        data['verification_level'] = self.verification_level.value
        return data


class MultiModalPillRecognizer:
    """
    Advanced pill recognizer with 5-step verification.
    
    Safety-first approach:
    1. Visual AI (CNN) - appearance matching
    2. Imprint OCR - most reliable identifier
    3. Physical size - dimension verification
    4. Database lookup - cross-reference
    5. User confirmation - final safety check
    """
    
    def __init__(self, visual_model=None, ocr_model=None, database=None):
        """
        Initialize recognizer with models.
        
        Args:
            visual_model: CNN model for visual recognition
            ocr_model: OCR model for imprint reading
            database: Pill database service
        """
        self.visual_model = visual_model
        self.ocr_model = ocr_model
        self.database = database
        
        # Verification weights (totals 100%)
        self.weights = {
            'imprint': 0.40,    # Most reliable!
            'visual': 0.25,
            'size': 0.20,
            'database': 0.15
        }
    
    async def recognize_pill(
        self,
        image: Image.Image,
        features: PillFeatures,
        user_context: Optional[Dict] = None
    ) -> PillRecognitionResult:
        """
        Main recognition pipeline with full verification.
        
        Args:
            image: PIL Image of pill
            features: Extracted physical features
            user_context: Optional user info (medications, history)
            
        Returns:
            Complete recognition result with all verifications
        """
        start_time = datetime.utcnow()
        
        # Step 1: Visual Recognition (CNN)
        visual_result = await self._verify_visual(image, features)
        candidates = visual_result.details.get('candidates', [])
        
        # Step 2: Imprint Verification (OCR) - CRITICAL!
        imprint_result = None
        if features.has_imprint and features.imprint_text:
            imprint_result = await self._verify_imprint(
                features.imprint_text,
                candidates
            )
        
        # Step 3: Size Verification
        size_result = None
        if features.diameter_mm or features.length_mm:
            size_result = await self._verify_size(
                features,
                candidates[0] if candidates else None
            )
        
        # Step 4: Database Cross-Reference
        database_result = await self._verify_database(
            features,
            candidates,
            user_context
        )
        
        # Calculate overall confidence
        confidence, level, passed_count = self._calculate_confidence([
            visual_result,
            imprint_result,
            size_result,
            database_result
        ])
        
        # Detect similar medications (confusion risk)
        similar_meds = await self._find_similar_medications(
            features,
            candidates[0] if candidates else None
        )
        
        # Generate safety warnings
        warnings, critical_warning = self._generate_warnings(
            confidence,
            level,
            visual_result,
            imprint_result,
            similar_meds
        )
        
        # Determine if pharmacist consultation required
        requires_pharmacist = (
            level in [VerificationLevel.LOW, VerificationLevel.UNVERIFIED, VerificationLevel.CRITICAL]
            or critical_warning is not None
            or confidence < 0.6
        )
        
        # Processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return PillRecognitionResult(
            medication_name=candidates[0]['name'] if candidates else 'Unknown',
            medication_id=candidates[0]['id'] if candidates else None,
            generic_name=candidates[0].get('generic_name') if candidates else None,
            dosage=candidates[0].get('dosage') if candidates else None,
            visual_verification=visual_result,
            imprint_verification=imprint_result,
            size_verification=size_result,
            database_verification=database_result,
            overall_confidence=confidence,
            verification_level=level,
            verifications_passed=passed_count,
            verifications_total=4,
            similar_medications=similar_meds,
            warnings=warnings,
            critical_warning=critical_warning,
            requires_pharmacist=requires_pharmacist,
            processing_time_ms=processing_time
        )
    
    async def _verify_visual(
        self,
        image: Image.Image,
        features: PillFeatures
    ) -> VerificationResult:
        """Step 1: Visual CNN-based recognition."""
        
        if not self.visual_model:
            return VerificationResult(
                method='visual_recognition',
                passed=False,
                confidence=0.0,
                details={'error': 'Visual model not available'}
            )
        
        # Get CNN predictions
        predictions = await self._cnn_predict(image)
        
        # Filter and rank by feature matching
        scored_predictions = []
        for pred in predictions:
            score = pred['confidence'] * 0.45  # Base CNN confidence
            
            # Shape matching (30% weight)
            if pred.get('shape') == features.shape:
                score += 0.30
            
            # Color matching (25% weight)
            if pred.get('color') == features.color_primary:
                score += 0.25
            
            pred['match_score'] = min(score, 1.0)
            scored_predictions.append(pred)
        
        # Sort by match score
        scored_predictions.sort(key=lambda x: x['match_score'], reverse=True)
        
        top_match = scored_predictions[0] if scored_predictions else None
        passed = top_match and top_match['match_score'] >= 0.60
        
        # Warning if similar medications
        warning = None
        if scored_predictions and len(scored_predictions) > 1:
            second_best = scored_predictions[1]
            if second_best['match_score'] > 0.50:
                warning = f"⚠️ Visually similar to {second_best['name']}"
        
        return VerificationResult(
            method='visual_recognition',
            passed=passed,
            confidence=top_match['match_score'] if top_match else 0.0,
            details={
                'candidates': scored_predictions[:5],
                'detected_shape': features.shape,
                'detected_color': features.color_primary
            },
            warning=warning
        )
    
    async def _verify_imprint(
        self,
        imprint_text: str,
        candidates: List[Dict]
    ) -> VerificationResult:
        """
        Step 2: Imprint code verification - MOST RELIABLE!
        
        Each pill has unique imprint code (e.g., "APO 500", "P 500").
        This is the gold standard for identification.
        """
        
        if not self.database:
            return VerificationResult(
                method='imprint_verification',
                passed=False,
                confidence=0.0,
                details={'error': 'Database not available'}
            )
        
        # Query database by imprint
        imprint_matches = await self.database.find_by_imprint(imprint_text)
        
        if not imprint_matches:
            return VerificationResult(
                method='imprint_verification',
                passed=False,
                confidence=0.1,
                details={
                    'imprint': imprint_text,
                    'matches_found': 0
                },
                warning=f"⚠️ Imprint '{imprint_text}' not found in database"
            )
        
        # Check if top candidate matches imprint
        if not candidates:
            # Only imprint data available
            return VerificationResult(
                method='imprint_verification',
                passed=True,
                confidence=0.90,
                details={
                    'imprint': imprint_text,
                    'matches': imprint_matches
                }
            )
        
        top_candidate = candidates[0]
        matches_top = any(
            match['id'] == top_candidate['id']
            for match in imprint_matches
        )
        
        # CRITICAL: Imprint mismatch!
        warning = None
        if not matches_top and imprint_matches:
            warning = (
                f"🚨 CRITICAL: Imprint '{imprint_text}' belongs to "
                f"{imprint_matches[0]['name']}, NOT {top_candidate['name']}!"
            )
        
        return VerificationResult(
            method='imprint_verification',
            passed=matches_top,
            confidence=0.95 if matches_top else 0.05,
            details={
                'imprint': imprint_text,
                'expected': top_candidate.get('imprint'),
                'matches': imprint_matches,
                'critical_mismatch': not matches_top and len(imprint_matches) > 0
            },
            warning=warning
        )
    
    async def _verify_size(
        self,
        features: PillFeatures,
        candidate: Optional[Dict]
    ) -> VerificationResult:
        """Step 3: Physical size verification."""
        
        if not candidate or not self.database:
            return VerificationResult(
                method='size_verification',
                passed=True,  # Skip if no data
                confidence=0.5,
                details={'note': 'Insufficient data for size verification'}
            )
        
        # Get expected size from database
        expected = await self.database.get_pill_dimensions(candidate['id'])
        
        if not expected:
            return VerificationResult(
                method='size_verification',
                passed=True,
                confidence=0.5,
                details={'note': 'Size data not available in database'}
            )
        
        # Calculate size difference
        measured = features.diameter_mm or features.length_mm
        expected_size = expected.get('diameter') or expected.get('length')
        
        if not measured or not expected_size:
            return VerificationResult(
                method='size_verification',
                passed=True,
                confidence=0.5,
                details={'note': 'Size measurement incomplete'}
            )
        
        # Tolerance: ±2mm
        size_diff = abs(measured - expected_size)
        tolerance = 2.0
        
        passed = size_diff <= tolerance
        confidence = max(0.0, 1.0 - (size_diff / 5.0))
        
        warning = None
        if not passed:
            warning = (
                f"⚠️ Size mismatch: Expected ~{expected_size}mm, "
                f"measured ~{measured}mm (±{size_diff:.1f}mm)"
            )
        
        return VerificationResult(
            method='size_verification',
            passed=passed,
            confidence=confidence,
            details={
                'measured_mm': measured,
                'expected_mm': expected_size,
                'difference_mm': size_diff,
                'tolerance_mm': tolerance
            },
            warning=warning
        )
    
    async def _verify_database(
        self,
        features: PillFeatures,
        candidates: List[Dict],
        user_context: Optional[Dict]
    ) -> VerificationResult:
        """Step 4: Cross-reference with comprehensive pill database."""
        
        if not self.database:
            return VerificationResult(
                method='database_verification',
                passed=False,
                confidence=0.0,
                details={'error': 'Database not available'}
            )
        
        # Search database by features
        db_results = await self.database.search_by_features(
            shape=features.shape,
            color=features.color_primary,
            imprint=features.imprint_text
        )
        
        if not db_results:
            return VerificationResult(
                method='database_verification',
                passed=False,
                confidence=0.2,
                details={'matches': 0},
                warning="⚠️ No matching pills found in database"
            )
        
        # Check if top candidate is in results
        confidence = 0.5
        passed = False
        
        if candidates:
            top_id = candidates[0]['id']
            passed = any(r['id'] == top_id for r in db_results)
            confidence = 0.75 if passed else 0.25
            
            # Boost confidence if in user's medication list
            if user_context and user_context.get('medications'):
                user_meds = user_context['medications']
                if candidates[0]['name'] in user_meds:
                    confidence += 0.20
        
        return VerificationResult(
            method='database_verification',
            passed=passed,
            confidence=min(confidence, 1.0),
            details={
                'database_matches': len(db_results),
                'top_matches': db_results[:3]
            }
        )
    
    async def _find_similar_medications(
        self,
        features: PillFeatures,
        top_candidate: Optional[Dict]
    ) -> List[Dict]:
        """Find visually similar medications (confusion risk)."""
        
        if not self.database:
            return []
        
        similar = await self.database.find_similar(
            shape=features.shape,
            color=features.color_primary,
            exclude_id=top_candidate['id'] if top_candidate else None,
            limit=5
        )
        
        return similar
    
    def _calculate_confidence(
        self,
        results: List[Optional[VerificationResult]]
    ) -> Tuple[float, VerificationLevel, int]:
        """
        Calculate weighted overall confidence.
        
        Returns:
            (confidence_score, verification_level, verifications_passed)
        """
        
        total_confidence = 0.0
        passed_count = 0
        has_critical_failure = False
        
        for result in results:
            if result is None:
                continue
            
            # Apply weight
            weight = self.weights.get(
                result.method.replace('_verification', '').replace('_recognition', ''),
                0.0
            )
            total_confidence += result.confidence * weight
            
            if result.passed:
                passed_count += 1
            
            # Check for critical imprint mismatch
            if result.method == 'imprint_verification' and not result.passed:
                if result.details.get('critical_mismatch'):
                    has_critical_failure = True
        
        # Determine verification level
        if has_critical_failure:
            level = VerificationLevel.CRITICAL
        elif passed_count >= 4:
            level = VerificationLevel.HIGH
        elif passed_count >= 2:
            level = VerificationLevel.MEDIUM
        elif passed_count >= 1:
            level = VerificationLevel.LOW
        else:
            level = VerificationLevel.UNVERIFIED
        
        return total_confidence, level, passed_count
    
    def _generate_warnings(
        self,
        confidence: float,
        level: VerificationLevel,
        visual_result: VerificationResult,
        imprint_result: Optional[VerificationResult],
        similar_meds: List[Dict]
    ) -> Tuple[List[str], Optional[str]]:
        """Generate user-facing safety warnings."""
        
        warnings = []
        critical_warning = None
        
        # Critical: Imprint mismatch
        if imprint_result and imprint_result.details.get('critical_mismatch'):
            critical_warning = (
                "🚨 DANGER: Pill imprint does NOT match the visual identification! "
                "This may be a completely different medication. "
                "DO NOT take this pill. Consult a pharmacist immediately."
            )
            warnings.append(critical_warning)
        
        # High risk: Low confidence
        elif confidence < 0.5:
            warnings.append(
                "⚠️ LOW CONFIDENCE: Cannot reliably identify this medication. "
                "Pharmacist verification REQUIRED before taking."
            )
        
        # Medium risk: No imprint verification
        if not imprint_result or not imprint_result.passed:
            if not critical_warning:  # Don't duplicate
                warnings.append(
                    "⚠️ Pill imprint not verified. Identification less reliable. "
                    "Check pill packaging or consult pharmacist."
                )
        
        # Confusion risk: Similar medications
        if similar_meds and len(similar_meds) >= 2:
            similar_names = ', '.join([m['name'] for m in similar_meds[:3]])
            warnings.append(
                f"⚠️ CAUTION: This pill looks similar to: {similar_names}. "
                "Verify the pill's imprint code carefully!"
            )
        
        # Low verification level
        if level == VerificationLevel.LOW:
            warnings.append(
                "⚠️ Only basic verification passed. "
                "Additional confirmation recommended."
            )
        elif level == VerificationLevel.UNVERIFIED:
            warnings.append(
                "❌ Medication could not be verified. "
                "Do not take without pharmacist confirmation."
            )
        
        return warnings, critical_warning
    
    async def _cnn_predict(self, image: Image.Image) -> List[Dict]:
        """Run CNN visual prediction (placeholder for actual model)."""
        
        if not self.visual_model:
            return []
        
        # This would call the actual CNN model
        # For now, return empty list
        return []
    
    def request_user_confirmation(
        self,
        result: PillRecognitionResult
    ) -> Dict[str, Any]:
        """
        Generate user confirmation questions.
        
        Returns prompts for user to manually verify pill features.
        """
        
        questions = []
        
        # Imprint confirmation (most important!)
        if result.imprint_verification:
            imprint = result.imprint_verification.details.get('imprint')
            if imprint:
                questions.append({
                    'type': 'imprint_check',
                    'question': f"Does your pill have '{imprint}' printed on it?",
                    'critical': True,
                    'weight': 0.40
                })
        
        # Shape confirmation
        if result.visual_verification:
            shape = result.visual_verification.details.get('detected_shape')
            questions.append({
                'type': 'shape_check',
                'question': f"Is your pill {shape} in shape?",
                'critical': False,
                'weight': 0.20
            })
        
        # Color confirmation
        color = result.visual_verification.details.get('detected_color')
        questions.append({
            'type': 'color_check',
            'question': f"Is your pill {color} in color?",
            'critical': False,
            'weight': 0.20
        })
        
        # Similar medication warning
        if result.similar_medications:
            questions.append({
                'type': 'confusion_check',
                'question': (
                    f"This pill looks similar to {result.similar_medications[0]['name']}. "
                    "Are you sure it's " + result.medication_name + "?"
                ),
                'critical': True,
                'weight': 0.20
            })
        
        return {
            'requires_confirmation': result.verification_level in [
                VerificationLevel.LOW,
                VerificationLevel.MEDIUM,
                VerificationLevel.CRITICAL
            ],
            'questions': questions,
            'recommendation': self._get_recommendation(result)
        }
    
    def _get_recommendation(self, result: PillRecognitionResult) -> str:
        """Get safety recommendation based on verification level."""
        
        if result.verification_level == VerificationLevel.CRITICAL:
            return (
                "🚨 DO NOT TAKE THIS PILL. Critical verification failure detected. "
                "Consult pharmacist immediately."
            )
        elif result.verification_level == VerificationLevel.UNVERIFIED:
            return (
                "❌ Cannot verify medication identity. Do not take without "
                "professional pharmacist confirmation."
            )
        elif result.verification_level == VerificationLevel.LOW:
            return (
                "⚠️ Low confidence identification. Verify pill imprint and "
                "packaging before taking. Pharmacist consultation recommended."
            )
        elif result.verification_level == VerificationLevel.MEDIUM:
            return (
                "✓ Moderate confidence. Double-check pill imprint matches "
                "the identified medication before taking."
            )
        else:  # HIGH
            return (
                "✓✓ High confidence identification. Pill features match database. "
                "Safe to proceed if packaging also matches."
            )
