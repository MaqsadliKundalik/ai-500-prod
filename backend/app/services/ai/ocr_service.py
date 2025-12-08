"""
OCR Service for Pill Imprint Recognition
=========================================
Extracts text from pill images (imprint codes)
"""

from typing import Optional, List, Dict, Tuple
from PIL import Image
import numpy as np
from dataclasses import dataclass


@dataclass
class OCRResult:
    """Result from OCR processing."""
    text: str
    confidence: float
    bounding_boxes: List[Tuple[int, int, int, int]]
    processing_method: str


class PillOCRService:
    """
    OCR service for reading imprint codes on pills.
    
    Supports multiple OCR backends:
    - EasyOCR (lightweight, good for pills)
    - Tesseract (fallback)
    - PaddleOCR (high accuracy)
    """
    
    def __init__(self, backend: str = 'easyocr'):
        """
        Initialize OCR service.
        
        Args:
            backend: OCR backend to use ('easyocr', 'tesseract', 'paddleocr')
        """
        self.backend = backend
        self.reader = None
        
        try:
            if backend == 'easyocr':
                self._init_easyocr()
            elif backend == 'tesseract':
                self._init_tesseract()
            elif backend == 'paddleocr':
                self._init_paddleocr()
        except Exception as e:
            print(f"⚠️ OCR backend {backend} not available: {e}")
            self.reader = None
    
    def _init_easyocr(self):
        """Initialize EasyOCR (recommended for pills)."""
        try:
            import easyocr
            # English and numbers only (pills typically use these)
            self.reader = easyocr.Reader(['en'], gpu=False)
        except ImportError:
            raise ImportError("EasyOCR not installed. Install: pip install easyocr")
    
    def _init_tesseract(self):
        """Initialize Tesseract OCR."""
        try:
            import pytesseract
            self.reader = 'tesseract'
        except ImportError:
            raise ImportError("Tesseract not installed. Install: pip install pytesseract")
    
    def _init_paddleocr(self):
        """Initialize PaddleOCR."""
        try:
            from paddleocr import PaddleOCR
            self.reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        except ImportError:
            raise ImportError("PaddleOCR not installed. Install: pip install paddleocr")
    
    async def extract_imprint(self, image: Image.Image) -> Optional[OCRResult]:
        """
        Extract imprint code from pill image.
        
        Args:
            image: PIL Image of pill
            
        Returns:
            OCRResult with extracted text
        """
        
        if not self.reader:
            return None
        
        # Preprocess image for better OCR
        processed_image = self._preprocess_for_ocr(image)
        
        # Run OCR based on backend
        if self.backend == 'easyocr':
            result = await self._ocr_easyocr(processed_image)
        elif self.backend == 'tesseract':
            result = await self._ocr_tesseract(processed_image)
        elif self.backend == 'paddleocr':
            result = await self._ocr_paddleocr(processed_image)
        else:
            return None
        
        # Post-process: clean up text
        if result:
            result.text = self._clean_imprint_text(result.text)
        
        return result
    
    def _preprocess_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Preprocess pill image for better OCR accuracy.
        
        Steps:
        1. Convert to grayscale
        2. Increase contrast
        3. Apply sharpening
        4. Resize for optimal OCR
        """
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to good OCR size (width ~800px)
        if image.width < 600:
            scale = 800 / image.width
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Enhance contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Sharpen
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        
        return image
    
    async def _ocr_easyocr(self, image: Image.Image) -> Optional[OCRResult]:
        """Run EasyOCR."""
        
        # Convert PIL to numpy
        img_array = np.array(image)
        
        # Run OCR
        results = self.reader.readtext(img_array)
        
        if not results:
            return None
        
        # Combine all detected text
        texts = []
        confidences = []
        boxes = []
        
        for bbox, text, conf in results:
            texts.append(text)
            confidences.append(conf)
            boxes.append(self._bbox_to_tuple(bbox))
        
        combined_text = ' '.join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return OCRResult(
            text=combined_text,
            confidence=avg_confidence,
            bounding_boxes=boxes,
            processing_method='easyocr'
        )
    
    async def _ocr_tesseract(self, image: Image.Image) -> Optional[OCRResult]:
        """Run Tesseract OCR."""
        
        import pytesseract
        from pytesseract import Output
        
        # Configure Tesseract for small text
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        # Run OCR
        data = pytesseract.image_to_data(
            image,
            config=custom_config,
            output_type=Output.DICT
        )
        
        # Extract text with confidence > 50
        texts = []
        boxes = []
        confidences = []
        
        for i, conf in enumerate(data['conf']):
            if int(conf) > 50:
                text = data['text'][i]
                if text.strip():
                    texts.append(text)
                    confidences.append(int(conf) / 100.0)
                    
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    boxes.append((x, y, x + w, y + h))
        
        if not texts:
            return None
        
        combined_text = ' '.join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return OCRResult(
            text=combined_text,
            confidence=avg_confidence,
            bounding_boxes=boxes,
            processing_method='tesseract'
        )
    
    async def _ocr_paddleocr(self, image: Image.Image) -> Optional[OCRResult]:
        """Run PaddleOCR."""
        
        # Convert PIL to numpy
        img_array = np.array(image)
        
        # Run OCR
        result = self.reader.ocr(img_array, cls=True)
        
        if not result or not result[0]:
            return None
        
        # Extract text
        texts = []
        confidences = []
        boxes = []
        
        for line in result[0]:
            bbox, (text, conf) = line
            texts.append(text)
            confidences.append(conf)
            boxes.append(self._bbox_to_tuple(bbox))
        
        combined_text = ' '.join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return OCRResult(
            text=combined_text,
            confidence=avg_confidence,
            bounding_boxes=boxes,
            processing_method='paddleocr'
        )
    
    def _clean_imprint_text(self, text: str) -> str:
        """
        Clean OCR output for imprint codes.
        
        Common OCR errors:
        - "O" vs "0" (letter O vs zero)
        - "I" vs "1" vs "l" (letter I vs one vs lowercase L)
        - Extra spaces
        """
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Common substitutions for pill imprints
        # (only apply if context suggests it's a number)
        cleaned = text.upper()
        
        # Remove common OCR artifacts
        cleaned = cleaned.replace('|', 'I')
        cleaned = cleaned.replace('[', '')
        cleaned = cleaned.replace(']', '')
        
        return cleaned.strip()
    
    def _bbox_to_tuple(self, bbox) -> Tuple[int, int, int, int]:
        """Convert bounding box to (x1, y1, x2, y2) tuple."""
        
        if isinstance(bbox[0], (list, tuple)):
            # Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        else:
            # Already a tuple
            return tuple(map(int, bbox))
    
    def validate_imprint_format(self, text: str) -> bool:
        """
        Validate if extracted text looks like a valid pill imprint.
        
        Valid formats:
        - "APO 500"
        - "P 500"
        - "M357"
        - "54 543"
        - etc.
        """
        
        if not text or len(text) > 20:  # Imprints are typically short
            return False
        
        # Must contain at least one alphanumeric character
        if not any(c.isalnum() for c in text):
            return False
        
        # Should not be too long
        words = text.split()
        if len(words) > 5:  # Most imprints are 1-3 words
            return False
        
        return True
