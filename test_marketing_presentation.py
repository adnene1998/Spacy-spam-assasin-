#!/usr/bin/env python3
"""
Test script for the marketing presentation generator.
Verifies that the presentation is created successfully.
"""

import os
import sys
from create_marketing_presentation import create_marketing_presentation

def test_presentation_creation():
    """Test that the presentation is created successfully"""
    print("Testing marketing presentation creation...")
    
    # Remove existing file if present
    filename = "Plan_Marketing_Plateforme_Gestion_Fetes.pptx"
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Removed existing file: {filename}")
    
    # Create the presentation
    try:
        result_filename = create_marketing_presentation()
        print(f"✓ Presentation created: {result_filename}")
    except Exception as e:
        print(f"✗ Failed to create presentation: {e}")
        sys.exit(1)
    
    # Verify file exists
    if not os.path.exists(filename):
        print(f"✗ Presentation file not found: {filename}")
        sys.exit(1)
    
    print(f"✓ Presentation file exists: {filename}")
    
    # Verify file size
    file_size = os.path.getsize(filename)
    if file_size < 1000:  # Should be at least 1KB
        print(f"✗ Presentation file is too small: {file_size} bytes")
        sys.exit(1)
    
    print(f"✓ Presentation file size is valid: {file_size} bytes")
    
    # Verify it's a valid PowerPoint file (check magic bytes)
    with open(filename, 'rb') as f:
        magic_bytes = f.read(4)
        # PowerPoint files are ZIP archives, so they start with PK
        if magic_bytes[:2] != b'PK':
            print(f"✗ File doesn't appear to be a valid PowerPoint file")
            sys.exit(1)
    
    print("✓ File appears to be a valid PowerPoint archive")
    
    print("\n✓ All tests passed successfully!")
    return True

if __name__ == "__main__":
    test_presentation_creation()
