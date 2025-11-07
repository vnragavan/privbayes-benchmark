#!/usr/bin/env python3
"""
Quick test script to verify the repository setup works correctly.
Tests imports, data loading, and basic adapter instantiation.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, project_root)

print("="*60)
print("Testing PrivBayes Benchmark Setup")
print("="*60)

# Test 1: Check data file exists
print("\n[1/5] Checking data file...")
data_path = os.path.join(project_root, 'data', 'adult.csv')
if os.path.exists(data_path):
    print(f"✓ Data file found: {data_path}")
    # Check file size
    size = os.path.getsize(data_path) / (1024 * 1024)  # MB
    print(f"  File size: {size:.2f} MB")
else:
    print(f"✗ Data file not found: {data_path}")
    sys.exit(1)

# Test 2: Check external implementations
print("\n[2/5] Checking external implementations...")
enhanced_path = os.path.join(project_root, 'external', 'privbayes_enhanced.py')
if os.path.exists(enhanced_path):
    print(f"✓ Enhanced PrivBayes found: {enhanced_path}")
else:
    print(f"✗ Enhanced PrivBayes not found: {enhanced_path}")
    sys.exit(1)

# Test 3: Test basic imports
print("\n[3/5] Testing basic imports...")
try:
    import pandas as pd
    print("✓ pandas imported")
except ImportError as e:
    print(f"✗ pandas import failed: {e}")
    print("  Install with: pip install pandas")
    sys.exit(1)

try:
    import numpy as np
    print("✓ numpy imported")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")
    print("  Install with: pip install numpy")
    sys.exit(1)

# Test 4: Load and inspect data
print("\n[4/5] Loading and inspecting data...")
try:
    df = pd.read_csv(data_path, nrows=5)  # Just read first 5 rows for speed
    print(f"✓ Data loaded successfully")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Sample columns: {list(df.columns[:5])}")
    print(f"  Data types: {df.dtypes.value_counts().to_dict()}")
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    sys.exit(1)

# Test 5: Test adapter imports (without instantiating)
print("\n[5/5] Testing adapter imports...")
try:
    from pbbench.variants.pb_enhanced import EnhancedPrivBayesAdapter
    print("✓ Enhanced PrivBayes adapter imported")
except ImportError as e:
    print(f"✗ Enhanced adapter import failed: {e}")
    sys.exit(1)

try:
    from pbbench.variants.pb_synthcity import SynthcityPrivBayesAdapter
    print("✓ SynthCity PrivBayes adapter imported")
except ImportError as e:
    print(f"⚠ SynthCity adapter import failed (may need synthcity package): {e}")
    print("  Install with: pip install synthcity")

try:
    from pbbench.variants.pb_datasynthesizer import DPMMPrivBayesAdapter
    print("✓ DPMM PrivBayes adapter imported")
except ImportError as e:
    print(f"⚠ DPMM adapter import failed (may need dpmm package): {e}")
    print("  Install with: pip install dpmm")

try:
    from pbbench.enhanced_metrics import (
        comprehensive_tvd_metrics,
        comprehensive_mi_metrics,
    )
    print("✓ Enhanced metrics module imported")
except ImportError as e:
    print(f"✗ Enhanced metrics import failed: {e}")
    sys.exit(1)

# Test 6: Check external implementation import
print("\n[6/6] Testing external implementation import...")
try:
    from external.privbayes_enhanced import PrivBayesSynthesizerEnhanced
    print("✓ Enhanced PrivBayes implementation imported")
except ImportError as e:
    print(f"✗ Enhanced implementation import failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✓ All basic tests passed!")
print("="*60)
print("\nTo run the full comparison, install dependencies:")
print("  pip install -r requirements.txt")
print("\nThen run:")
print("  python3 scripts/comprehensive_comparison.py \\")
print("      --data data/adult.csv \\")
print("      --eps 1.0 \\")
print("      --seeds 0 \\")
print("      --out-dir test_results \\")
print("      --implementations Enhanced")
print("="*60)


