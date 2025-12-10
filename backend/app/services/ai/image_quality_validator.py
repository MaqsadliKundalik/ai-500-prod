"""
Image Quality Validator
========================
Validates image quality before processing for better recognition accuracy
"""

from typing import Dict, Tuple
from PIL import Image
import numpy as np
import cv2
from io import BytesIO


class ImageQualityValidator:
    """Validates image quality for pill recognition."""
    
    def __init__(
        self,
        min_resolution: Tuple[int, int] = (200, 200),
        max_resolution: Tuple[int, int] = (4096, 4096),
        min_brightness: float = 20.0,
        max_brightness: float = 235.0,
        blur_threshold: float = 100.0
    ):
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.blur_threshold = blur_threshold
    
    def validate(self, image: Image.Image) -> Dict:
        """
        Validate image quality.
        
        Returns:
            {
                'is_valid': bool,
                'issues': List[str],
                'metrics': {
                    'resolution': Tuple[int, int],
                    'brightness': float,
                    'blur_score': float,
                    'contrast': float
                },
                'suggestions': List[str]
            }
        """
        issues = []
        suggestions = []
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Check if grayscale or RGB
        if len(img_array.shape) == 2:
            gray = img_array
        else:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 1. Resolution check
        width, height = image.size
        if width < self.min_resolution[0] or height < self.min_resolution[1]:
            issues.append(f"Resolution too low: {width}x{height}")
            suggestions.append(f"Use image at least {self.min_resolution[0]}x{self.min_resolution[1]} pixels")
        
        if width > self.max_resolution[0] or height > self.max_resolution[1]:
            issues.append(f"Resolution too high: {width}x{height}")
            suggestions.append("Image will be resized automatically")
        
        # 2. Brightness check
        brightness = np.mean(gray)
        if brightness < self.min_brightness:
            issues.append(f"Image too dark: {brightness:.1f}")
            suggestions.append("Take photo in better lighting")
        elif brightness > self.max_brightness:
            issues.append(f"Image too bright: {brightness:.1f}")
            suggestions.append("Avoid direct flash or sunlight")
        
        # 3. Blur detection (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()
        
        if blur_score < self.blur_threshold:
            issues.append(f"Image blurry: {blur_score:.1f}")
            suggestions.append("Hold camera steady, ensure focus")
        
        # 4. Contrast check
        contrast = gray.std()
        if contrast < 20:
            issues.append(f"Low contrast: {contrast:.1f}")
            suggestions.append("Ensure pill is on contrasting background")
        
        # 5. Check if image is mostly one color (likely blank/invalid)
        unique_colors = len(np.unique(gray))
        if unique_colors < 10:
            issues.append("Image appears blank or uniform")
            suggestions.append("Ensure pill is visible in frame")
        
        metrics = {
            'resolution': (width, height),
            'brightness': float(brightness),
            'blur_score': float(blur_score),
            'contrast': float(contrast),
            'unique_colors': int(unique_colors)
        }
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'suggestions': suggestions,
            'metrics': metrics,
            'quality_score': self._calculate_quality_score(metrics)
        }
    
    def _calculate_quality_score(self, metrics: Dict) -> float:
        """Calculate overall quality score (0-100)."""
        score = 100.0
        
        # Resolution penalty
        width, height = metrics['resolution']
        if width < 400 or height < 400:
            score -= 20
        
        # Brightness penalty
        brightness = metrics['brightness']
        if brightness < 50 or brightness > 200:
            score -= 15
        
        # Blur penalty
        if metrics['blur_score'] < 100:
            score -= 25
        elif metrics['blur_score'] < 200:
            score -= 10
        
        # Contrast penalty
        if metrics['contrast'] < 30:
            score -= 15
        
        # Unique colors penalty
        if metrics['unique_colors'] < 50:
            score -= 15
        
        return max(0.0, score)
    
    def enhance_image(self, image: Image.Image) -> Image.Image:
        """
        Apply automatic enhancements to improve quality.
        
        - Adjust brightness/contrast
        - Sharpen if needed
        - Denoise
        """
        img_array = np.array(image)
        
        # Convert to LAB color space for better processing
        if len(img_array.shape) == 3:
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge back
            enhanced_lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img_array)
        
        # Denoise
        if len(enhanced.shape) == 3:
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        else:
            enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # Sharpen
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return Image.fromarray(enhanced)
    
    def get_quality_feedback(self, quality_result: Dict) -> str:
        """Get user-friendly quality feedback in Uzbek."""
        score = quality_result['quality_score']
        
        if score >= 80:
            return "✅ Surat sifati zo'r! Aniqlash boshlandi..."
        elif score >= 60:
            feedback = "⚠️ Surat sifati o'rtacha. "
            if quality_result['suggestions']:
                feedback += quality_result['suggestions'][0]
            return feedback
        else:
            feedback = "❌ Surat sifati past. Iltimos:\n"
            for suggestion in quality_result['suggestions'][:3]:
                feedback += f"• {suggestion}\n"
            return feedback


def get_image_quality_validator() -> ImageQualityValidator:
    """Get singleton instance of image quality validator."""
    return ImageQualityValidator()
