#!/usr/bin/env python3
"""
Test script to verify API key rotation functionality.
This script tests the key rotation logic without making actual API calls.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from nlp import get_key_status, reset_key_failures, rotate_to_next_key, GEMINI_API_KEYS

def test_key_rotation():
    """Test the key rotation functionality."""
    print("=== API Key Rotation Test ===")
    
    # Check if keys are loaded
    if not GEMINI_API_KEYS:
        print("❌ No API keys found!")
        return False
    
    print(f"✅ Loaded {len(GEMINI_API_KEYS)} API keys")
    
    # Get initial status
    status = get_key_status()
    print(f"✅ Current key index: {status['current_key_index']}")
    print(f"✅ Current key preview: {status['current_key_preview']}")
    
    # Test key rotation
    original_index = status['current_key_index']
    print(f"\n--- Testing manual key rotation ---")
    
    try:
        rotate_to_next_key()
        new_status = get_key_status()
        
        if new_status['current_key_index'] != original_index:
            print(f"✅ Key rotation successful: {original_index} -> {new_status['current_key_index']}")
        else:
            print("❌ Key rotation failed - index didn't change")
            return False
            
    except Exception as e:
        print(f"❌ Key rotation failed with error: {e}")
        return False
    
    # Test failure reset
    print(f"\n--- Testing failure reset ---")
    try:
        reset_key_failures()
        status_after_reset = get_key_status()
        print(f"✅ Failure reset successful - failed keys: {status_after_reset['failed_keys']}")
    except Exception as e:
        print(f"❌ Failure reset failed: {e}")
        return False
    
    # Display key status summary
    print(f"\n--- Key Status Summary ---")
    status = get_key_status()
    print(f"Total keys: {status['total_keys']}")
    print(f"Current key: {status['current_key_index']} ({status['current_key_preview']})")
    print(f"Failed keys: {status['failed_keys']}")
    
    # Show first few and last few keys
    print(f"\n--- Key Details (first 3 and last 3) ---")
    for i, key_info in enumerate(status['key_details']):
        if i < 3 or i >= len(status['key_details']) - 3:
            status_str = "🔴 FAILED" if key_info['is_failed'] else "🟢 OK"
            current_str = "◀ CURRENT" if key_info['is_current'] else ""
            print(f"Key {key_info['index']:2d}: {key_info['key_preview']} - {status_str} {current_str}")
    
    print(f"\n✅ All tests passed!")
    return True

def test_api_call_simulation():
    """Simulate API calls to test rotation under load."""
    print(f"\n=== Simulating API Calls ===")
    
    # Reset state
    reset_key_failures()
    
    # Simulate multiple API calls
    print("Simulating 10 API calls...")
    for i in range(10):
        try:
            # This would normally make an API call, but we'll just rotate
            rotate_to_next_key()
            status = get_key_status()
            print(f"Call {i+1}: Using key {status['current_key_index']} ({status['current_key_preview']})")
        except Exception as e:
            print(f"Call {i+1}: Failed - {e}")
            break
    
    print("✅ API call simulation completed")

if __name__ == "__main__":
    success = test_key_rotation()
    if success:
        test_api_call_simulation()
    else:
        sys.exit(1)
