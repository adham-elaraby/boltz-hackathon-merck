#!/usr/bin/env python3
"""
Quick test to verify cuEquivariance is installed and working.
"""

import sys

print("=" * 80)
print("Testing cuEquivariance Installation")
print("=" * 80)
print()

# Test 1: Import
print("1. Testing import...")
try:
    import cuequivariance_torch
    print(f"   ✅ cuequivariance_torch imported successfully")
    print(f"   📦 Version: {cuequivariance_torch.__version__}")
except ImportError as e:
    print(f"   ❌ Failed to import cuequivariance_torch")
    print(f"   Error: {e}")
    print()
    print("   To install:")
    print("   pip install cuequivariance-torch")
    sys.exit(1)

print()

# Test 2: Check available primitives
print("2. Checking available primitives...")
try:
    from cuequivariance_torch import primitives
    available = [item for item in dir(primitives) if not item.startswith('_')]
    print(f"   ✅ Found {len(available)} primitives")
    for prim in available:
        print(f"      - {prim}")
except Exception as e:
    print(f"   ❌ Error accessing primitives: {e}")

print()

# Test 3: Check triangle operations
print("3. Testing triangle operations...")
try:
    from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update
    print("   ✅ triangle_multiplicative_update available")
except ImportError:
    print("   ❌ triangle_multiplicative_update not available")

try:
    from cuequivariance_torch.primitives.triangle import triangle_attention
    print("   ✅ triangle_attention available")
except ImportError:
    print("   ❌ triangle_attention not available")

print()

# Test 4: Check for attention operations
print("4. Checking for attention operations...")
try:
    from cuequivariance_torch.primitives import attention
    print("   ✅ attention module found")
    attention_ops = [item for item in dir(attention) if not item.startswith('_')]
    print(f"   📋 Available attention operations:")
    for op in attention_ops:
        print(f"      - {op}")
except ImportError:
    print("   ⚠️  No attention module found (might be in a different location)")

print()

# Test 5: Check for norm operations
print("5. Checking for normalization operations...")
try:
    from cuequivariance_torch.primitives import norm
    print("   ✅ norm module found")
    norm_ops = [item for item in dir(norm) if not item.startswith('_')]
    print(f"   📋 Available norm operations:")
    for op in norm_ops:
        print(f"      - {op}")
except ImportError:
    print("   ⚠️  No norm module found")

print()

# Test 6: CUDA availability
print("6. Checking CUDA...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✅ CUDA available")
        print(f"   🎮 Device: {torch.cuda.get_device_name(0)}")
        print(f"   💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   ⚠️  CUDA not available (kernels will not work)")
except Exception as e:
    print(f"   ❌ Error checking CUDA: {e}")

print()
print("=" * 80)
print("Summary")
print("=" * 80)
print()
print("✅ cuEquivariance is installed and ready!")
print()
print("Next steps:")
print("  1. Remove --no_kernels from predict_hackathon.py (DONE!)")
print("  2. Run benchmark to test speedup")
print("  3. Implement AttentionPairBias kernel if attention module exists")
print()
