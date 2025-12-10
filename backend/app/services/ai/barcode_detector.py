"""
Barcode/QR Code Detection Service
==================================
Detects and decodes barcodes and QR codes from images using OpenCV and pyzbar
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import pyzbar, but make it optional for deployment
try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    logger.warning("pyzbar not available - barcode scanning will be disabled")
    PYZBAR_AVAILABLE = False
    pyzbar = None


class BarcodeDetector:
    """Service for detecting and decoding barcodes/QR codes from images."""
    
    SUPPORTED_TYPES = {
        'QRCODE': 'qr',
        'EAN13': 'ean13',
        'EAN8': 'ean8',
        'UPCA': 'upc_a',
        'UPCE': 'upc_e',
        'CODE128': 'code128',
        'CODE39': 'code39',
        'ITF': 'itf',
        'DATAMATRIX': 'datamatrix',
        'PDF417': 'pdf417'
    }
    
    def __init__(self):
        self.min_confidence = 0.5
    
    def detect_codes(self, image_data: bytes) -> List[Dict]:
        """
        Detect and decode all barcodes/QR codes in an image.
        
        Args:
            image_data: Raw image bytes
        
        Returns:
            List of detected codes with metadata
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("Failed to decode image")
                return []
            
            # Convert to grayscale for better detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Try multiple preprocessing techniques
            results = []
            
            # 1. Original grayscale
            codes = self._decode_from_image(gray)
            results.extend(codes)
            
            # 2. Thresholded image
            if not codes:
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                codes = self._decode_from_image(thresh)
                results.extend(codes)
            
            # 3. Contrast enhancement
            if not codes:
                enhanced = cv2.equalizeHist(gray)
                codes = self._decode_from_image(enhanced)
                results.extend(codes)
            
            # 4. Blurred (for noisy images)
            if not codes:
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                codes = self._decode_from_image(blurred)
                results.extend(codes)
            
            # Remove duplicates based on data and type
            unique_results = self._remove_duplicates(results)
            
            logger.info(f"Detected {len(unique_results)} unique codes")
            return unique_results
        
        except Exception as e:
            logger.error(f"Error detecting codes: {e}")
            return []
    
    def _decode_from_image(self, image: np.ndarray) -> List[Dict]:
        """Decode barcodes from a preprocessed image."""
        if not PYZBAR_AVAILABLE or pyzbar is None:
            logger.warning("pyzbar not available - cannot decode barcodes")
            return []
            
        decoded_objects = pyzbar.decode(image)
        results = []
        
        for obj in decoded_objects:
            try:
                # Decode data
                data = obj.data.decode('utf-8')
                
                # Get code type
                code_type = self.SUPPORTED_TYPES.get(
                    obj.type,
                    obj.type.lower()
                )
                
                # Get bounding box
                x, y, w, h = obj.rect
                
                result = {
                    'data': data,
                    'type': code_type,
                    'raw_type': obj.type,
                    'quality': obj.quality,
                    'bbox': {
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h
                    },
                    'polygon': [
                        {'x': point.x, 'y': point.y}
                        for point in obj.polygon
                    ]
                }
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error decoding object: {e}")
                continue
        
        return results
    
    def _remove_duplicates(self, codes: List[Dict]) -> List[Dict]:
        """Remove duplicate codes."""
        seen = set()
        unique = []
        
        for code in codes:
            key = (code['data'], code['type'])
            if key not in seen:
                seen.add(key)
                unique.append(code)
        
        return unique
    
    def detect_single_code(self, image_data: bytes, 
                          preferred_type: Optional[str] = None) -> Optional[Dict]:
        """
        Detect single barcode/QR code (returns first or preferred type).
        
        Args:
            image_data: Raw image bytes
            preferred_type: Preferred code type ('qr', 'ean13', etc.)
        
        Returns:
            Single detected code or None
        """
        codes = self.detect_codes(image_data)
        
        if not codes:
            return None
        
        # If preferred type specified, try to find it
        if preferred_type:
            for code in codes:
                if code['type'] == preferred_type:
                    return code
        
        # Otherwise return first code
        return codes[0]
    
    def validate_barcode(self, barcode: str, barcode_type: str) -> bool:
        """
        Validate barcode checksum.
        
        Args:
            barcode: Barcode string
            barcode_type: Type of barcode
        
        Returns:
            True if valid
        """
        if barcode_type == 'ean13':
            return self._validate_ean13(barcode)
        elif barcode_type == 'ean8':
            return self._validate_ean8(barcode)
        elif barcode_type == 'upc_a':
            return self._validate_upc_a(barcode)
        
        # For other types, assume valid
        return True
    
    def _validate_ean13(self, barcode: str) -> bool:
        """Validate EAN-13 checksum."""
        if len(barcode) != 13 or not barcode.isdigit():
            return False
        
        # Calculate checksum
        odd_sum = sum(int(barcode[i]) for i in range(0, 12, 2))
        even_sum = sum(int(barcode[i]) for i in range(1, 12, 2))
        checksum = (10 - ((odd_sum + even_sum * 3) % 10)) % 10
        
        return checksum == int(barcode[12])
    
    def _validate_ean8(self, barcode: str) -> bool:
        """Validate EAN-8 checksum."""
        if len(barcode) != 8 or not barcode.isdigit():
            return False
        
        # EAN-8: weights are 3,1,3,1,3,1,3,1 (odd positions get weight 3)
        odd_sum = sum(int(barcode[i]) for i in range(0, 7, 2))  # positions 1,3,5,7
        even_sum = sum(int(barcode[i]) for i in range(1, 7, 2))  # positions 2,4,6
        checksum = (10 - ((odd_sum * 3 + even_sum) % 10)) % 10
        
        return checksum == int(barcode[7])
    
    def _validate_upc_a(self, barcode: str) -> bool:
        """Validate UPC-A checksum."""
        if len(barcode) != 12 or not barcode.isdigit():
            return False
        
        odd_sum = sum(int(barcode[i]) for i in range(0, 11, 2))
        even_sum = sum(int(barcode[i]) for i in range(1, 11, 2))
        checksum = (10 - ((odd_sum * 3 + even_sum) % 10)) % 10
        
        return checksum == int(barcode[11])
    
    def get_barcode_info(self, barcode: str, barcode_type: str) -> Dict:
        """
        Get information about a barcode.
        
        Returns:
            Dictionary with barcode metadata
        """
        info = {
            'barcode': barcode,
            'type': barcode_type,
            'length': len(barcode),
            'is_valid': self.validate_barcode(barcode, barcode_type),
            'country_code': None,
            'manufacturer_code': None
        }
        
        # Extract country and manufacturer codes for EAN/UPC
        if barcode_type in ['ean13', 'ean8'] and barcode.isdigit():
            if barcode_type == 'ean13':
                info['country_code'] = barcode[:3]
                info['manufacturer_code'] = barcode[3:7]
                info['product_code'] = barcode[7:12]
                info['check_digit'] = barcode[12]
            elif barcode_type == 'ean8':
                info['country_code'] = barcode[:2]
                info['product_code'] = barcode[2:7]
                info['check_digit'] = barcode[7]
        
        return info


# Global instance
_barcode_detector = None


def get_barcode_detector() -> BarcodeDetector:
    """Get or create BarcodeDetector instance."""
    global _barcode_detector
    if _barcode_detector is None:
        _barcode_detector = BarcodeDetector()
    return _barcode_detector
