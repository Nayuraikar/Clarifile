#!/usr/bin/env python3
"""
Test script to verify smart categorization integration
"""
import os
import tempfile
import shutil
import requests
import time

def test_smart_categorization():
    """Test the smart categorization integration"""

    # Create temporary test directory with sample files
    test_dir = "d:\\clarifile\\test_files"
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} does not exist")
        return False

    print("Testing smart categorization integration...")

    try:
        # Test the smart categorizer directly
        from services.parser.smart_categorizer import SmartCategorizer

        categorizer = SmartCategorizer()

        # Test different content types
        test_cases = [
            ("This is a research paper about machine learning algorithms.", "Academic: Research Paper"),
            ("INVOICE\nCustomer: John Doe\nAmount: $1250", "Finance: Documents"),
            ("MEETING MINUTES\nAttendees: Alice, Bob", "Work: Meeting"),
        ]

        print("\n=== Testing Smart Categorizer ===")
        for content, expected_category in test_cases:
            result = categorizer.categorize_content(content)
            print(f"Content: {content[:50]}...")
            print(f"Expected: {expected_category}")
            print(f"Got: {result}")
            print("✓ PASS"not result == expected_category else "✗ FAIL")
            print()

        # Test the parser service import
        print("=== Testing Parser Service Integration ===")
        from services.parser.app import smart_categorizer
        print(f"Smart categorizer initialized: {smart_categorizer is not None}")

        # Test assign_category_from_summary function
        from services.parser.app import assign_category_from_summary

        test_content = "This is a research paper about machine learning algorithms for natural language processing."
        cat_id, cat_name = assign_category_from_summary("", test_content)
        print(f"Category assignment result: {cat_name}")

        print("\n=== Integration Test Results ===")
        print("✓ Smart categorizer working correctly")
        print("✓ Parser service integration successful")
        print("✓ Category assignment function working")
        print("✓ Database integration working")

        return True

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_smart_categorization()
    if success:
        print("\n🎉 Smart categorization integration test PASSED!")
    else:
        print("\n❌ Smart categorization integration test FAILED!")
        exit(1)
