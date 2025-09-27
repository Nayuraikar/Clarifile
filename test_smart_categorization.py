#!/usr/bin/env python3
"""
test_smart_categorization.py
Test script to demonstrate the smart categorization functionality.
"""

import os
import sys

# Add the services path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'parser'))

from smart_categorizer import SmartCategorizer
from nlp import classify_with_gemini

def test_smart_categorization():
    """Test the smart categorization functionality."""
    print("=== Testing Smart Categorization ===\n")

    # Initialize the categorizer
    categorizer = SmartCategorizer()

    # Test cases with different types of content
    test_cases = [
        {
            "name": "Academic Research Paper",
            "content": """
            Abstract
            This paper presents a novel approach to machine learning algorithms for natural language processing.
            We propose a new neural network architecture that improves upon existing transformer models.

            Introduction
            Natural language processing has seen significant advances in recent years with the introduction
            of transformer-based models. Our research focuses on improving the efficiency of these models
            while maintaining their performance on various NLP tasks.

            Methodology
            We conducted extensive experiments using benchmark datasets including GLUE and SuperGLUE.
            Our proposed model was trained on large-scale datasets using distributed computing techniques.

            Results
            Our experimental results show significant improvements over baseline models, with a 15% increase
            in F1-score on the MNLI task and 12% improvement on the SQuAD dataset.

            Conclusion
            This work demonstrates the effectiveness of our proposed approach and opens new directions
            for future research in efficient transformer architectures.
            """,
            "expected_keywords": ["academic", "research", "paper", "methodology", "results"]
        },
        {
            "name": "Financial Document",
            "content": """
            INVOICE

            Invoice Number: INV-2024-001
            Date: January 15, 2024

            Bill To:
            John Smith
            123 Business Ave
            New York, NY 10001

            Description: Consulting Services
            Amount: $5,000.00

            Subtotal: $5,000.00
            Tax (8%): $400.00
            Total: $5,400.00

            Payment Terms: Net 30 days
            Due Date: February 14, 2024

            Thank you for your business!
            """,
            "expected_keywords": ["invoice", "payment", "financial", "amount", "total"]
        },
        {
            "name": "Meeting Notes",
            "content": """
            MEETING NOTES - Project Kickoff

            Date: January 10, 2024
            Time: 2:00 PM - 4:00 PM
            Location: Conference Room A
            Attendees: Sarah Johnson, Mike Chen, Alex Rodriguez

            Agenda:
            1. Project objectives and scope
            2. Timeline and milestones
            3. Resource allocation
            4. Risk assessment

            Discussion Points:
            - Agreed on project scope: Develop new mobile application
            - Timeline: 6 months from kickoff to launch
            - Budget allocated: $150,000
            - Key risks identified: Technology dependencies, timeline slippage

            Action Items:
            1. Sarah to create detailed project plan by next Friday
            2. Mike to research technology stack options
            3. Alex to prepare budget breakdown

            Next Meeting: January 17, 2024 at 2:00 PM
            """,
            "expected_keywords": ["meeting", "agenda", "attendees", "action items"]
        }
    ]

    print("Testing individual content categorization:")
    print("-" * 50)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}:")
        print("   Content preview:", test_case['content'][:100] + "...")

        # Test smart categorizer
        category = categorizer.categorize_content(test_case['content'])
        print(f"   Smart Category: {category}")

        # Test integrated classification
        integrated_category = classify_with_gemini(test_case['content'])
        print(f"   Integrated Category: {integrated_category}")

        # Check if expected keywords are present
        content_lower = test_case['content'].lower()
        found_keywords = [kw for kw in test_case['expected_keywords'] if kw in content_lower]
        print(f"   Expected keywords found: {found_keywords}")

    print("\n" + "=" * 50)
    print("Smart categorization integration test completed!")
    print("\nThe system now uses:")
    print("1. Smart categorization with embeddings and clustering (primary)")
    print("2. Gemini AI classification (fallback)")
    print("3. Keyword-based classification (final fallback)")
    print("\nThis provides intelligent, content-based categorization without")
    print("requiring paid API services for the core functionality.")

if __name__ == "__main__":
    test_smart_categorization()
