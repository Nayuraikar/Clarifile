#!/usr/bin/env python3
"""
Test the enhanced smart categorization with real content analysis
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'parser'))

def test_enhanced_categorization():
    """Test the enhanced smart categorization"""
    
    try:
        from smart_categorizer import SmartCategorizer
        
        categorizer = SmartCategorizer()
        
        # Test cases with actual content that should be properly categorized
        test_cases = [
            {
                "content": """INVOICE
Invoice Number: INV-2024-001
Date: January 15, 2024
Bill To: John Doe Company
123 Business Street
Amount Due: $1,250.00
Description: Software development services
Payment Terms: Net 30 days
Total: $1,250.00""",
                "expected_category": "Finance"
            },
            {
                "content": """MEETING MINUTES
Date: January 15, 2024
Time: 2:00 PM - 3:00 PM
Attendees: Alice Johnson, Bob Smith, Charlie Brown
Agenda:
1. Project timeline review
2. Budget allocation discussion
3. Resource planning
4. Next steps

Action Items:
- Alice to prepare budget report by next week
- Bob to update project timeline
- Charlie to coordinate with external vendors

Next meeting: January 22, 2024""",
                "expected_category": "Work"
            },
            {
                "content": """Abstract
This research paper presents a comprehensive analysis of machine learning algorithms for natural language processing tasks. We examine various transformer architectures including BERT, GPT, and T5, evaluating their performance on text classification, sentiment analysis, and named entity recognition.

Introduction
Natural language processing (NLP) has seen significant advances with the introduction of transformer-based models. This study aims to provide empirical evidence for the effectiveness of different approaches.

Methodology
We conducted experiments using standard benchmark datasets including GLUE, SuperGLUE, and CoNLL-2003. Our experimental setup involved fine-tuning pre-trained models on downstream tasks.

Results
Our findings indicate that transformer models achieve state-of-the-art performance across multiple NLP tasks, with BERT showing particular strength in classification tasks.

Conclusion
This research contributes to the understanding of transformer architectures in NLP applications and provides guidelines for practitioners.""",
                "expected_category": "Academic"
            },
            {
                "content": """SOFTWARE LICENSE AGREEMENT

This Software License Agreement ("Agreement") is entered into between TechCorp Inc. ("Licensor") and the end user ("Licensee").

1. GRANT OF LICENSE
Subject to the terms and conditions of this Agreement, Licensor grants Licensee a non-exclusive, non-transferable license to use the software.

2. RESTRICTIONS
Licensee shall not:
a) Modify, adapt, or create derivative works
b) Reverse engineer, decompile, or disassemble
c) Distribute or sublicense the software

3. TERM AND TERMINATION
This Agreement is effective until terminated. Licensor may terminate this Agreement immediately upon breach.

4. LIABILITY
IN NO EVENT SHALL LICENSOR BE LIABLE FOR ANY DAMAGES ARISING FROM USE OF THE SOFTWARE.""",
                "expected_category": "Legal"
            }
        ]
        
        print("🧠 Testing Enhanced Smart Categorization")
        print("=" * 50)
        
        all_passed = True
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}:")
            print(f"Content preview: {test_case['content'][:100]}...")
            
            result = categorizer.categorize_content(test_case['content'])
            expected = test_case['expected_category']
            
            print(f"Expected category type: {expected}")
            print(f"Actual result: {result}")
            
            # Check if the result contains the expected category type
            if expected.lower() in result.lower():
                print("✅ PASS - Correct category type detected")
            else:
                print("❌ FAIL - Incorrect categorization")
                all_passed = False
            
            print("-" * 30)
        
        print(f"\n🎯 Overall Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_categorization()
    if success:
        print("\n🎉 Enhanced smart categorization is working correctly!")
        print("The system now uses transformer-based semantic analysis for accurate content categorization.")
    else:
        print("\n⚠️  There are issues with the enhanced categorization.")
        exit(1)
