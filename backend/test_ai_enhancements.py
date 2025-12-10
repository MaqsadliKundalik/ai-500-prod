"""
Quick Test - AI Enhancements
============================
Test all new features
"""

def test_image_quality_validator():
    """Test image quality validation."""
    print("\n" + "="*60)
    print("TEST 1: Image Quality Validator")
    print("="*60)
    
    from app.services.ai.image_quality_validator import get_image_quality_validator
    from PIL import Image
    import numpy as np
    
    validator = get_image_quality_validator()
    
    # Create test images
    # 1. Good quality image
    good_image = Image.fromarray(np.random.randint(50, 200, (500, 500, 3), dtype=np.uint8))
    result = validator.validate(good_image)
    print(f"\nGood Image: Score = {result['quality_score']:.1f}/100")
    print(f"Valid: {result['is_valid']}")
    
    # 2. Too small
    small_image = Image.fromarray(np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8))
    result = validator.validate(small_image)
    print(f"\nSmall Image: Score = {result['quality_score']:.1f}/100")
    print(f"Issues: {result['issues']}")
    
    # 3. Too dark
    dark_image = Image.fromarray(np.random.randint(0, 30, (500, 500, 3), dtype=np.uint8))
    result = validator.validate(dark_image)
    print(f"\nDark Image: Score = {result['quality_score']:.1f}/100")
    print(f"Feedback (UZ): {validator.get_quality_feedback(result)}")
    
    print("\n✅ Image Quality Validator working!")


def test_drug_interaction_explainer():
    """Test drug interaction explanations in Uzbek."""
    print("\n" + "="*60)
    print("TEST 2: Drug Interaction Explainer (Uzbek)")
    print("="*60)
    
    from app.services.ai.drug_interaction_explainer import get_drug_interaction_explainer
    
    explainer = get_drug_interaction_explainer()
    
    # Test severity levels
    print("\nSeverity Levels:")
    for severity in ["none", "mild", "moderate", "severe", "fatal"]:
        info = explainer.get_severity_info(severity)
        print(f"{info['emoji']} {info['name']}: {info['description']}")
    
    # Test patient explanation
    print("\nPatient Explanation:")
    explanation = explainer.generate_patient_explanation(
        "Aspirin",
        "Warfarin",
        "severe",
        "pharmacodynamic",
        "Ikki dori ham qonni suyultiradi - qon ketish xavfi oshadi"
    )
    print(explanation)
    
    # Test monitoring recommendations
    recommendations = explainer.get_monitoring_recommendations(
        "severe",
        ["cardiovascular", "hepatic"]
    )
    print("\nMonitoring Recommendations:")
    for rec in recommendations:
        print(f"• {rec}")
    
    print("\n✅ Drug Interaction Explainer working!")


def test_uzbek_nlu():
    """Test Uzbek NLU engine."""
    print("\n" + "="*60)
    print("TEST 3: Uzbek NLU Engine")
    print("="*60)
    
    from app.services.ai.uzbek_nlu_engine import get_uzbek_nlu_engine
    
    nlu = get_uzbek_nlu_engine()
    
    test_queries = [
        "Aspirin dori bor mi?",
        "Yaqin dorixona qayerda?",
        "Paracetamol va Ibuprofen birgalikda ichsa bo'ladimi?",
        "Citramon narxi qancha?",
        "Bosh og'rig'i uchun dori",
        "Здравствуйте, где аптека?",
    ]
    
    for query in test_queries:
        result = nlu.classify_intent(query)
        print(f"\nQuery: {query}")
        print(f"Language: {result['language']}")
        print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
        if result['entities']:
            print(f"Entities: {result['entities']}")
        
        # Get response template
        template = nlu.get_response_template(result['intent'], result['language'])
        print(f"Response: {template}")
    
    print("\n✅ Uzbek NLU Engine working!")


def test_pharmacy_enhancements():
    """Test pharmacy enhancements (without database)."""
    print("\n" + "="*60)
    print("TEST 4: Pharmacy Enhancements")
    print("="*60)
    
    from app.services.ai.pharmacy_enhancements import PharmacyEnhancements
    
    # Test Haversine distance
    enh = PharmacyEnhancements(None)
    
    # Tashkent center coordinates
    tashkent_center = (41.2995, 69.2401)
    # Another location ~5km away
    location_2 = (41.3295, 69.2701)
    
    distance = enh._haversine_distance(
        tashkent_center[0], tashkent_center[1],
        location_2[0], location_2[1]
    )
    
    print(f"\nDistance calculation:")
    print(f"From: {tashkent_center}")
    print(f"To: {location_2}")
    print(f"Distance: {distance:.2f} km")
    
    # Test route optimization (mock data)
    user_location = tashkent_center
    pharmacies = [
        {"id": "1", "name": "Dorixona 1", "latitude": 41.31, "longitude": 69.25, "available_medications": ["med1", "med2"]},
        {"id": "2", "name": "Dorixona 2", "latitude": 41.32, "longitude": 69.27, "available_medications": ["med3"]},
        {"id": "3", "name": "Dorixona 3", "latitude": 41.28, "longitude": 69.23, "available_medications": ["med1"]},
    ]
    
    route = enh.calculate_route_optimization(
        user_location,
        pharmacies.copy(),
        ["med1", "med2", "med3"]
    )
    
    print(f"\nRoute Optimization:")
    print(f"Total pharmacies to visit: {route['total_pharmacies']}")
    print(f"Total distance: {route['total_distance_km']} km")
    print(f"Estimated time: {route['estimated_time_minutes']} minutes")
    print(f"Medications found: {route['medications_found']}/{route['medications_found'] + len(route['medications_not_found'])}")
    
    print("\n✅ Pharmacy Enhancements working!")


def test_batch_recall_checker():
    """Test batch recall checker."""
    print("\n" + "="*60)
    print("TEST 5: Batch Recall Checker")
    print("="*60)
    
    from app.services.ai.batch_recall_checker import get_batch_recall_checker
    
    checker = get_batch_recall_checker()
    
    # Test risk level calculation
    test_recalls = [
        {"classification": "Class II", "severity": "MEDIUM"},
        {"classification": "Class III", "severity": "LOW"},
    ]
    
    risk_level = checker._calculate_risk_level(test_recalls)
    action = checker._get_action_required(risk_level, test_recalls)
    
    print(f"\nRisk Level: {risk_level}")
    print(f"Action Required:\n{action}")
    
    # Test classification translation
    translations = [
        "Class I",
        "Class II",
        "Class III",
    ]
    
    print("\nFDA Classification Translations:")
    for classification in translations:
        uz_text = checker._translate_classification_uz(classification)
        print(f"{classification} → {uz_text}")
    
    print("\n✅ Batch Recall Checker working!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("AI ENHANCEMENTS - QUICK TESTS")
    print("="*60)
    
    try:
        test_image_quality_validator()
    except Exception as e:
        print(f"❌ Image Quality Validator test failed: {e}")
    
    try:
        test_drug_interaction_explainer()
    except Exception as e:
        print(f"❌ Drug Interaction Explainer test failed: {e}")
    
    try:
        test_uzbek_nlu()
    except Exception as e:
        print(f"❌ Uzbek NLU test failed: {e}")
    
    try:
        test_pharmacy_enhancements()
    except Exception as e:
        print(f"❌ Pharmacy Enhancements test failed: {e}")
    
    try:
        test_batch_recall_checker()
    except Exception as e:
        print(f"❌ Batch Recall Checker test failed: {e}")
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED!")
    print("="*60)
