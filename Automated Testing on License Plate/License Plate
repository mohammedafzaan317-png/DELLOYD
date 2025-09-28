#!/usr/bin/python
"""
Complete License Plate Testing Solution - All in One File
Enhanced with multiple license plate comparisons and detailed testing
"""

import difflib
import pytest
import random
import string
from typing import List, Tuple, Dict
import numpy as np

# String Comparison Functions
def align_strings(s1: str, s2: str):
    """Align two strings using difflib"""
    matcher = difflib.SequenceMatcher(None, s1, s2)
    aligned1, aligned2 = [], []
    i1 = i2 = 0

    for block in matcher.get_matching_blocks():
        a1, a2, length = block
        while i1 < a1 or i2 < a2:
            if (a1 - i1) > (a2 - i2):
                aligned1.append(s1[i1])
                aligned2.append('-')
                i1 += 1
            elif (a2 - i2) > (a1 - i1):
                aligned1.append('-')
                aligned2.append(s2[i2])
                i2 += 1
            else:
                aligned1.append(s1[i1])
                aligned2.append(s2[i2])
                i1 += 1
                i2 += 1

        for k in range(length):
            aligned1.append(s1[a1 + k])
            aligned2.append(s2[a2 + k])
        i1, i2 = a1 + length, a2 + length

    return ''.join(aligned1), ''.join(aligned2)

def compute_similarity(al1: str, al2: str):
    """Compute percentage similarity between aligned strings"""
    assert len(al1) == len(al2)
    matches = total = 0
    for c1, c2 in zip(al1, al2):
        if c1 == '-' or c2 == '-':
            continue
        total += 1
        if c1 == c2:
            matches += 1
    return 100.0 * matches / total if total > 0 else 0.0

def generate_detailed_report(al1: str, al2: str):
    """Generate detailed alignment report"""
    report = []
    report.append("STRING ALIGNMENT REPORT")
    report.append("=" * 50)
    report.append(f"Aligned String 1: {al1}")
    report.append(f"Aligned String 2: {al2}")
    
    # Create matching indicators
    match_line = "Comparison:    "
    for c1, c2 in zip(al1, al2):
        if c1 == c2 and c1 != '-':
            match_line += "|"
        elif c1 == '-' or c2 == '-':
            match_line += " "
        else:
            match_line += "X"
    
    report.append(match_line)
    report.append("")
    
    # Detailed position analysis
    report.append("POSITION-WISE ANALYSIS:")
    report.append("-" * 30)
    
    for idx, (c1, c2) in enumerate(zip(al1, al2)):
        pos = idx + 1
        if c1 == c2:
            report.append(f"Position {pos:2d}: '{c1}' = '{c2}' → MATCH")
        elif c1 == '-':
            report.append(f"Position {pos:2d}: GAP   vs '{c2}' → INSERTION")
        elif c2 == '-':
            report.append(f"Position {pos:2d}: '{c1}' vs GAP    → DELETION")
        else:
            report.append(f"Position {pos:2d}: '{c1}' vs '{c2}' → MISMATCH")
    
    return '\n'.join(report)

def compare_strings(s1: str, s2: str):
    """Main comparison function with detailed reporting"""
    if not (6 <= len(s1) <= 10 and 6 <= len(s2) <= 10):
        raise ValueError("Strings must be 6 to 10 characters long")
    
    al1, al2 = align_strings(s1, s2)
    similarity = compute_similarity(al1, al2)
    report = generate_detailed_report(al1, al2)
    
    return similarity, report

# License Plate Generator Class
class IndianLicensePlateGenerator:
    """Generate valid and invalid Indian license plate strings"""
    
    # Indian state codes
    STATE_CODES = ['AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JH', 
                  'KA', 'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OD', 'PB', 
                  'RJ', 'SK', 'TN', 'TS', 'TR', 'UP', 'UK', 'WB', 'DL', 'PY']
    
    # Series letters
    SERIES_LETTERS = ['A', 'B', 'C', 'AB', 'CD', 'EF', 'GH', 'IJ', 'KL', 'MN']
    
    @staticmethod
    def generate_valid_plate():
        """Generate a valid Indian license plate"""
        state = random.choice(IndianLicensePlateGenerator.STATE_CODES)
        district = f"{random.randint(1, 99):02d}"
        series = random.choice(IndianLicensePlateGenerator.SERIES_LETTERS)
        number = f"{random.randint(1, 9999):04d}"
        
        formats = [
            f"{state}{district}{series}{number}",      # KA01AB1234
            f"{state}-{district}-{series}-{number}",   # KA-01-AB-1234
            f"{state} {district} {series} {number}",   # KA 01 AB 1234
        ]
        return random.choice(formats)
    
    @staticmethod
    def generate_invalid_plate():
        """Generate an invalid license plate"""
        invalid_types = [
            lambda: ''.join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(3, 5))),
            lambda: ''.join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(11, 15))),
            lambda: ''.join(random.choices('!@#$%^&*()', k=random.randint(6, 10))),
            lambda: f"{random.randint(1000, 9999)}{random.choice(IndianLicensePlateGenerator.STATE_CODES)}",
        ]
        return random.choice(invalid_types)()

# License Plate Comparison Functions
def compare_license_plates(plate1: str, plate2: str, description: str = ""):
    """Compare two license plates and print detailed results"""
    print(f"\n🔍 COMPARISON: {description}")
    print("=" * 60)
    print(f"Plate 1: {plate1}")
    print(f"Plate 2: {plate2}")
    
    try:
        similarity, report = compare_strings(plate1, plate2)
        print(f"📊 Similarity Score: {similarity:.2f}%")
        
        # Interpretation of similarity
        if similarity == 100:
            interpretation = "✅ IDENTICAL PLATES"
        elif similarity >= 80:
            interpretation = "✅ VERY SIMILAR (likely same vehicle)"
        elif similarity >= 60:
            interpretation = "⚠️  MODERATELY SIMILAR (possible match)"
        elif similarity >= 40:
            interpretation = "⚠️  SLIGHTLY SIMILAR (unlikely match)"
        else:
            interpretation = "❌ VERY DIFFERENT (different vehicles)"
        
        print(f"🎯 Interpretation: {interpretation}")
        print("\n" + report)
        
        return similarity
        
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 0

def run_comprehensive_license_plate_comparisons():
    """Run 4 detailed license plate comparisons with different scenarios"""
    
    generator = IndianLicensePlateGenerator()
    
    print("🚗 LICENSE PLATE COMPARISON ANALYSIS")
    print("=" * 70)
    
    # Comparison 1: Identical plates
    plate1 = "KA01AB1234"
    plate2 = "KA01AB1234"
    compare_license_plates(plate1, plate2, "Identical Plates")
    
    # Comparison 2: Same state, different numbers
    plate1 = "KA01AB1234"
    plate2 = "KA01AB5678"
    compare_license_plates(plate1, plate2, "Same State/District, Different Numbers")
    
    # Comparison 3: Different state, same numbers
    plate1 = "KA01AB1234"
    plate2 = "MH01AB1234"
    compare_license_plates(plate1, plate2, "Different State, Same Numbers")
    
    # Comparison 4: Completely different plates
    plate1 = "KA01AB1234"
    plate2 = "DL12CD5678"
    compare_license_plates(plate1, plate2, "Completely Different Plates")
    
    # Comparison 5: Format variations (with spaces/hyphens)
    plate1 = "KA01AB1234"
    plate2 = "KA-01-AB-1234"
    compare_license_plates(plate1, plate2, "Format Variations")
    
    # Comparison 6: Similar but different series
    plate1 = "KA01AB1234"
    plate2 = "KA01CD1234"
    compare_license_plates(plate1, plate2, "Different Series Letters")
    
    # Comparison 7: One character difference
    plate1 = "KA01AB1234"
    plate2 = "KA01AB1235"
    compare_license_plates(plate1, plate2, "Single Character Difference")
    
    # Comparison 8: Different length handling
    plate1 = "KA01AB123"
    plate2 = "KA01AB1234"
    compare_license_plates(plate1, plate2, "Different Length Plates")

def generate_test_cases():
    """Generate test cases for automated testing"""
    test_cases = [
        # (plate1, plate2, expected_min_similarity, description)
        ("KA01AB1234", "KA01AB1234", 100, "Identical plates"),
        ("KA01AB1234", "KA01AB5678", 80, "Same format, different numbers"),
        ("KA01AB1234", "MH01AB1234", 60, "Different state codes"),
        ("KA01AB1234", "KA01CD1234", 70, "Different series letters"),
        ("KA01AB1234", "KA-01-AB-1234", 85, "Format variations"),
        ("KA01AB1234", "DL12XY9999", 30, "Completely different"),
    ]
    return test_cases

def run_automated_tests():
    """Run automated tests on license plate comparisons"""
    print("\n" + "=" * 70)
    print("🤖 AUTOMATED TESTING RESULTS")
    print("=" * 70)
    
    test_cases = generate_test_cases()
    results = []
    
    for plate1, plate2, expected_min, description in test_cases:
        try:
            similarity, _ = compare_strings(plate1, plate2)
            passed = similarity >= expected_min
            status = "✅ PASS" if passed else "❌ FAIL"
            
            results.append({
                'description': description,
                'plate1': plate1,
                'plate2': plate2,
                'similarity': similarity,
                'expected_min': expected_min,
                'passed': passed
            })
            
            print(f"{status} {description}")
            print(f"   Plates: '{plate1}' vs '{plate2}'")
            print(f"   Similarity: {similarity:.1f}% (Expected ≥ {expected_min}%)")
            print()
            
        except ValueError as e:
            print(f"❌ ERROR: {description} - {e}")
            results.append({
                'description': description,
                'error': str(e),
                'passed': False
            })
    
    # Summary statistics
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.get('passed', False))
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print("📊 TEST SUMMARY")
    print("-" * 30)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    return results

def performance_analysis():
    """Analyze performance on bulk comparisons"""
    print("\n" + "=" * 70)
    print("⚡ PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    generator = IndianLicensePlateGenerator()
    test_plates = [generator.generate_valid_plate() for _ in range(100)]
    
    comparisons = 0
    total_similarity = 0
    
    # Compare each plate with next one
    for i in range(len(test_plates) - 1):
        try:
            similarity, _ = compare_strings(test_plates[i], test_plates[i+1])
            total_similarity += similarity
            comparisons += 1
        except ValueError:
            continue
    
    if comparisons > 0:
        avg_similarity = total_similarity / comparisons
        print(f"Average similarity across {comparisons} random plate comparisons: {avg_similarity:.2f}%")
        
        if avg_similarity < 40:
            print("✅ Good discrimination between different plates")
        elif avg_similarity < 60:
            print("⚠️  Moderate discrimination")
        else:
            print("❌ Poor discrimination - many similar plates detected")
    
    return avg_similarity if comparisons > 0 else 0

if __name__ == "__main__":
    # Run the comprehensive comparisons
    run_comprehensive_license_plate_comparisons()
    
    # Run automated tests
    test_results = run_automated_tests()
    
    # Performance analysis
    performance_analysis()
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎯 LICENSE PLATE COMPARISON SYSTEM - COMPLETE")
    print("=" * 70)
    print("This system provides:")
    print("✅ String alignment and similarity scoring")
    print("✅ Detailed comparison reports")
    print("✅ Automated testing framework")
    print("✅ Performance analysis")
    print("✅ Support for Indian license plate formats")
    print("\nUse cases:")
    print("• Vehicle identification and matching")
    print("• License plate validation systems")
    print("• Traffic monitoring and analysis")
    print("• Database deduplication")
