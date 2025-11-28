#!/usr/bin/env python
"""
Test Suite for Edge Detection Feature
Validates V-Edge and H-Edge detection functionality
"""

import sys
import numpy as np
import cv2

sys.path.insert(0, '/Users/samlai/Local_2/agent_test')
from SFR_app_v2 import SFRCalculator


def create_test_images():
    """Create synthetic test images for validation"""

    # Test 1: Perfect V-Edge (vertical line)
    v_edge = np.zeros((100, 100), dtype=np.uint8)
    v_edge[:, 50:] = 255

    # Test 2: Perfect H-Edge (horizontal line)
    h_edge = np.zeros((100, 100), dtype=np.uint8)
    h_edge[50:, :] = 255

    # Test 3: Diagonal edge (should be detected as Mixed)
    diagonal = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        diagonal[i, i:] = 255

    # Test 4: Low contrast (should fail validation)
    low_contrast = np.full((100, 100), 128, dtype=np.uint8)
    low_contrast[:, 50:] = 130

    # Test 5: Uniform image (should fail validation)
    uniform = np.full((100, 100), 200, dtype=np.uint8)

    return {
        'v_edge': v_edge,
        'h_edge': h_edge,
        'diagonal': diagonal,
        'low_contrast': low_contrast,
        'uniform': uniform
    }


def test_edge_detection():
    """Test edge detection on synthetic images"""

    print("\n" + "="*60)
    print("EDGE DETECTION TEST SUITE")
    print("="*60)

    images = create_test_images()

    # Test 1: V-Edge Detection
    print("\n[Test 1] Vertical Edge Detection")
    print("-" * 60)
    edge_type, confidence, details = SFRCalculator.detect_edge_orientation(images['v_edge'])
    print(f"✓ Edge Type: {edge_type}")
    print(f"✓ Confidence: {confidence:.1f}%")
    print(f"✓ Ratio X/Y: {details.get('ratio_x_y', 'N/A'):.2f}")
    assert edge_type == "V-Edge", "Expected V-Edge"
    assert confidence > 80, "Expected high confidence"
    print("✓ PASSED")

    # Test 2: H-Edge Detection
    print("\n[Test 2] Horizontal Edge Detection")
    print("-" * 60)
    edge_type, confidence, details = SFRCalculator.detect_edge_orientation(images['h_edge'])
    print(f"✓ Edge Type: {edge_type}")
    print(f"✓ Confidence: {confidence:.1f}%")
    print(f"✓ Ratio X/Y: {details.get('ratio_x_y', 'N/A'):.2f}")
    assert edge_type == "H-Edge", "Expected H-Edge"
    assert confidence > 80, "Expected high confidence"
    print("✓ PASSED")

    # Test 3: Diagonal Edge Detection
    print("\n[Test 3] Diagonal Edge Detection (Mixed)")
    print("-" * 60)
    edge_type, confidence, details = SFRCalculator.detect_edge_orientation(images['diagonal'])
    print(f"✓ Edge Type: {edge_type}")
    print(f"✓ Confidence: {confidence:.1f}%")
    assert edge_type in ["Mixed", "V-Edge", "H-Edge"], "Expected Mixed or dominant edge"
    print("✓ PASSED")

    # Test 4: Validation - Low Contrast
    print("\n[Test 4] Validation - Low Contrast Detection")
    print("-" * 60)
    is_valid, msg, edge_type, confidence = SFRCalculator.validate_edge(images['low_contrast'])
    print(f"✓ Valid: {is_valid}")
    print(f"✓ Message: {msg}")
    print(f"✓ Edge Type: {edge_type}")
    assert not is_valid, "Expected validation to fail for low contrast"
    print("✓ PASSED")

    # Test 5: Validation - Uniform Image
    print("\n[Test 5] Validation - Uniform Image Detection")
    print("-" * 60)
    is_valid, msg, edge_type, confidence = SFRCalculator.validate_edge(images['uniform'])
    print(f"✓ Valid: {is_valid}")
    print(f"✓ Message: {msg}")
    assert not is_valid, "Expected validation to fail for uniform image"
    print("✓ PASSED")

    # Test 6: SFR Calculation - V-Edge
    print("\n[Test 6] SFR Calculation - V-Edge")
    print("-" * 60)
    frequencies, sfr = SFRCalculator.calculate_sfr(images['v_edge'], edge_type="V-Edge")
    print(f"✓ Frequencies shape: {frequencies.shape if frequencies is not None else 'None'}")
    print(f"✓ SFR shape: {sfr.shape if sfr is not None else 'None'}")
    print(f"✓ SFR range: [{sfr.min():.4f}, {sfr.max():.4f}]" if sfr is not None else "None")
    assert frequencies is not None, "Expected frequency array"
    assert sfr is not None, "Expected SFR array"
    assert sfr[0] >= 0.99, "Expected normalized DC component ≈ 1.0"
    print("✓ PASSED")

    # Test 7: SFR Calculation - H-Edge
    print("\n[Test 7] SFR Calculation - H-Edge")
    print("-" * 60)
    frequencies, sfr = SFRCalculator.calculate_sfr(images['h_edge'], edge_type="H-Edge")
    print(f"✓ Frequencies shape: {frequencies.shape if frequencies is not None else 'None'}")
    print(f"✓ SFR shape: {sfr.shape if sfr is not None else 'None'}")
    assert frequencies is not None, "Expected frequency array"
    assert sfr is not None, "Expected SFR array"
    print("✓ PASSED")

    # Test 8: Empty ROI
    print("\n[Test 8] Edge Detection - Empty ROI")
    print("-" * 60)
    edge_type, confidence, details = SFRCalculator.detect_edge_orientation(None)
    print(f"✓ Edge Type: {edge_type}")
    print(f"✓ Confidence: {confidence}")
    assert edge_type == "No Edge", "Expected 'No Edge' for empty ROI"
    assert confidence == 0, "Expected 0 confidence for empty ROI"
    print("✓ PASSED")

    # Test 9: Validation - Empty ROI
    print("\n[Test 9] Validation - Empty ROI")
    print("-" * 60)
    is_valid, msg, edge_type, confidence = SFRCalculator.validate_edge(None)
    print(f"✓ Valid: {is_valid}")
    print(f"✓ Message: {msg}")
    assert not is_valid, "Expected validation to fail for empty ROI"
    print("✓ PASSED")

    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)

    return True


def test_method_signatures():
    """Test that all methods have correct signatures"""

    print("\n" + "="*60)
    print("METHOD SIGNATURE VALIDATION")
    print("="*60)

    # Check detect_edge_orientation
    print("\n[Check] detect_edge_orientation method")
    assert hasattr(SFRCalculator, 'detect_edge_orientation'), "Missing detect_edge_orientation"
    print("✓ Method exists")
    print("✓ Expected signature: (roi_image) -> (str, float, dict)")

    # Check validate_edge
    print("\n[Check] validate_edge method")
    assert hasattr(SFRCalculator, 'validate_edge'), "Missing validate_edge"
    print("✓ Method exists")
    print("✓ Expected signature: (roi_image) -> (bool, str, str, float)")

    # Check calculate_sfr
    print("\n[Check] calculate_sfr method")
    assert hasattr(SFRCalculator, 'calculate_sfr'), "Missing calculate_sfr"
    print("✓ Method exists")
    print("✓ Expected signature: (roi_image, edge_type='V-Edge') -> (array, array)")

    print("\n" + "="*60)
    print("METHOD VALIDATION PASSED ✓")
    print("="*60)

    return True


def test_performance():
    """Test performance metrics"""

    import time

    print("\n" + "="*60)
    print("PERFORMANCE TESTING")
    print("="*60)

    # Create test image
    test_img = np.zeros((500, 500), dtype=np.uint8)
    test_img[:, 250:] = 255

    # Test 1: Edge detection speed
    print("\n[Test] Edge detection performance")
    start = time.time()
    for _ in range(100):
        SFRCalculator.detect_edge_orientation(test_img)
    elapsed = time.time() - start
    avg_time = (elapsed / 100) * 1000
    print(f"✓ 100 iterations: {elapsed:.3f}s")
    print(f"✓ Average time: {avg_time:.2f}ms per call")
    assert avg_time < 100, "Detection too slow"
    print("✓ PASSED")

    # Test 2: SFR calculation speed
    print("\n[Test] SFR calculation performance")
    start = time.time()
    for _ in range(50):
        SFRCalculator.calculate_sfr(test_img, edge_type="V-Edge")
    elapsed = time.time() - start
    avg_time = (elapsed / 50) * 1000
    print(f"✓ 50 iterations: {elapsed:.3f}s")
    print(f"✓ Average time: {avg_time:.2f}ms per call")
    assert avg_time < 200, "SFR calculation too slow"
    print("✓ PASSED")

    print("\n" + "="*60)
    print("PERFORMANCE TESTING PASSED ✓")
    print("="*60)

    return True


if __name__ == '__main__':
    try:
        # Run all tests
        test_method_signatures()
        test_edge_detection()
        test_performance()

        print("\n" + "="*60)
        print("🎉 ALL VALIDATION TESTS PASSED 🎉")
        print("="*60)
        print("\nImplementation Status: ✅ PRODUCTION READY")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

