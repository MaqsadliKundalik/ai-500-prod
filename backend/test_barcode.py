"""
Test Barcode Detection
======================
"""

from app.services.ai.barcode_detector import get_barcode_detector
import qrcode
from PIL import Image, ImageDraw, ImageFont
import io


def generate_test_qr():
    """Generate test QR code."""
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data("MEDICATION-ID-12345")
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


def generate_test_barcode():
    """Generate simple test barcode image (EAN-13)."""
    # Create image with text (simulating barcode)
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw vertical lines (simple barcode simulation)
    ean13_code = "4600051000057"  # Valid EAN-13
    
    x = 50
    for i, digit in enumerate(ean13_code):
        # Alternate thick/thin lines
        width = 3 if int(digit) % 2 == 0 else 1
        draw.rectangle([x, 50, x + width, 150], fill='black')
        x += 5
    
    # Add text below
    try:
        draw.text((100, 160), ean13_code, fill='black')
    except:
        pass
    
    # Convert to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


def test_qr_detection():
    """Test QR code detection."""
    print("\n" + "="*60)
    print("TEST 1: QR Code Detection")
    print("="*60)
    
    detector = get_barcode_detector()
    
    # Generate test QR
    qr_image = generate_test_qr()
    print(f"Generated QR code image ({len(qr_image)} bytes)")
    
    # Detect
    codes = detector.detect_codes(qr_image)
    
    if codes:
        print(f"\n✓ Detected {len(codes)} code(s)")
        for i, code in enumerate(codes, 1):
            print(f"\nCode {i}:")
            print(f"  Type: {code['type']}")
            print(f"  Data: {code['data']}")
            print(f"  Quality: {code.get('quality', 'N/A')}")
            print(f"  BBox: {code.get('bbox')}")
    else:
        print("\n✗ No codes detected")


def test_barcode_validation():
    """Test barcode validation."""
    print("\n" + "="*60)
    print("TEST 2: Barcode Validation")
    print("="*60)
    
    detector = get_barcode_detector()
    
    test_cases = [
        ("4600051000057", "ean13", True),   # Valid EAN-13
        ("1234567890128", "ean13", True),   # Valid EAN-13
        ("1234567890123", "ean13", False),  # Invalid checksum
        ("12345670", "ean8", True),         # Valid EAN-8
        ("12345679", "ean8", False),        # Invalid checksum
    ]
    
    for barcode, barcode_type, expected_valid in test_cases:
        is_valid = detector.validate_barcode(barcode, barcode_type)
        status = "✓" if is_valid == expected_valid else "✗"
        print(f"\n{status} {barcode} ({barcode_type})")
        print(f"   Expected: {expected_valid}, Got: {is_valid}")


def test_barcode_info():
    """Test barcode info extraction."""
    print("\n" + "="*60)
    print("TEST 3: Barcode Info Extraction")
    print("="*60)
    
    detector = get_barcode_detector()
    
    barcodes = [
        ("4600051000057", "ean13"),  # Russian product
        ("5901234123457", "ean13"),  # Polish product
        ("12345670", "ean8"),        # Valid EAN-8
    ]
    
    for barcode, barcode_type in barcodes:
        info = detector.get_barcode_info(barcode, barcode_type)
        print(f"\n{barcode} ({barcode_type}):")
        print(f"  Valid: {info['is_valid']}")
        if info.get('country_code'):
            print(f"  Country Code: {info['country_code']}")
        if info.get('manufacturer_code'):
            print(f"  Manufacturer: {info['manufacturer_code']}")
        if info.get('product_code'):
            print(f"  Product: {info['product_code']}")


def test_supported_types():
    """Show supported barcode types."""
    print("\n" + "="*60)
    print("TEST 4: Supported Barcode Types")
    print("="*60)
    
    detector = get_barcode_detector()
    
    print("\nSupported barcode/QR code types:")
    for raw_type, normalized_type in detector.SUPPORTED_TYPES.items():
        print(f"  - {raw_type}: {normalized_type}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("BARCODE/QR CODE DETECTION TESTS")
    print("="*60)
    
    test_qr_detection()
    test_barcode_validation()
    test_barcode_info()
    test_supported_types()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
