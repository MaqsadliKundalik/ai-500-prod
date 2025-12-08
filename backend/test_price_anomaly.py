"""
Test Price Anomaly Detection
=============================
"""

from app.services.ai.price_anomaly_service import get_price_anomaly_service


def test_normal_price():
    """Test detection with normal price."""
    print("\n" + "="*60)
    print("TEST 1: Normal Price")
    print("="*60)
    
    service = get_price_anomaly_service()
    
    result = service.detect_anomaly(
        medicine_name="Paracetamol 500mg",
        region="Tashkent",
        pharmacy="Remedy",
        current_price=15000,
        inn="paracetamol",
        atx_code="N02BE01",
        base_price=15000
    )
    
    print(f"Medicine: {result['medicine_name']}")
    print(f"Price: {result['current_price']} UZS")
    print(f"Base Price: {result['base_price']} UZS")
    print(f"Deviation: {result['deviation_percent']:.2f}%")
    print(f"Is Anomaly: {result['is_anomaly']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Severity: {result['severity']}")
    print(f"\nMethods:")
    for method, data in result['methods'].items():
        print(f"  {method}: {data}")


def test_overpriced():
    """Test detection with overpriced medicine."""
    print("\n" + "="*60)
    print("TEST 2: Overpriced Medicine (2x normal)")
    print("="*60)
    
    service = get_price_anomaly_service()
    
    result = service.detect_anomaly(
        medicine_name="Paracetamol 500mg",
        region="Tashkent",
        pharmacy="Unknown Pharmacy",
        current_price=30000,  # 2x the normal price
        inn="paracetamol",
        atx_code="N02BE01",
        base_price=15000
    )
    
    print(f"Medicine: {result['medicine_name']}")
    print(f"Price: {result['current_price']} UZS")
    print(f"Base Price: {result['base_price']} UZS")
    print(f"Deviation: {result['deviation_percent']:.2f}%")
    print(f"Is Anomaly: {result['is_anomaly']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Severity: {result['severity']}")


def test_underpriced():
    """Test detection with suspiciously cheap medicine."""
    print("\n" + "="*60)
    print("TEST 3: Underpriced Medicine (50% off - suspicious)")
    print("="*60)
    
    service = get_price_anomaly_service()
    
    result = service.detect_anomaly(
        medicine_name="Paracetamol 500mg",
        region="Tashkent",
        pharmacy="Cheap Pharmacy",
        current_price=7500,  # 50% off
        inn="paracetamol",
        atx_code="N02BE01",
        base_price=15000
    )
    
    print(f"Medicine: {result['medicine_name']}")
    print(f"Price: {result['current_price']} UZS")
    print(f"Base Price: {result['base_price']} UZS")
    print(f"Deviation: {result['deviation_percent']:.2f}%")
    print(f"Is Anomaly: {result['is_anomaly']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Severity: {result['severity']}")


def test_regional_comparison():
    """Test regional price comparison."""
    print("\n" + "="*60)
    print("TEST 4: Regional Price Comparison")
    print("="*60)
    
    service = get_price_anomaly_service()
    
    result = service.compare_regional_prices(
        medicine_name="Paracetamol 500mg",
        inn="paracetamol",
        atx_code="N02BE01",
        base_price=15000
    )
    
    print(f"Medicine: {result['medicine_name']}")
    print(f"Base Price: {result['base_price']} UZS")
    print(f"\nRegional Price Estimates:")
    
    for region, data in result['regional_prices'].items():
        print(f"\n  {region}:")
        print(f"    Expected: {data['expected_price']} UZS")
        print(f"    Range: {data['price_range']['min']} - {data['price_range']['max']} UZS")


def test_expensive_medicine():
    """Test with expensive medicine (100K+ UZS)."""
    print("\n" + "="*60)
    print("TEST 5: Expensive Medicine")
    print("="*60)
    
    service = get_price_anomaly_service()
    
    result = service.detect_anomaly(
        medicine_name="Atsiklovir Infuziya",
        region="Tashkent",
        pharmacy="Remedy",
        current_price=65000,
        inn="ацикловир",
        atx_code="J05AB01",
        base_price=65182
    )
    
    print(f"Medicine: {result['medicine_name']}")
    print(f"Price: {result['current_price']} UZS")
    print(f"Base Price: {result['base_price']} UZS")
    print(f"Deviation: {result['deviation_percent']:.2f}%")
    print(f"Is Anomaly: {result['is_anomaly']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Severity: {result['severity']}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PRICE ANOMALY DETECTION TESTS")
    print("="*60)
    
    test_normal_price()
    test_overpriced()
    test_underpriced()
    test_regional_comparison()
    test_expensive_medicine()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
