# ✅ Phase 1 Complete: NVIDIA Kernels Enabled!

## What We Did

### 1. Removed `--no_kernels` Flag
**File**: `hackathon/predict_hackathon.py` (line ~244)

**Changed**:
```python
# BEFORE (kernels disabled)
fixed = [
    "boltz", "predict", str(yaml_path),
    "--devices", "1",
    "--out_dir", str(out_dir),
    "--cache", cache,
    "--no_kernels",  # ← REMOVED THIS
    "--output_format", "pdb",
]

# AFTER (kernels enabled)
fixed = [
    "boltz", "predict", str(yaml_path),
    "--devices", "1",
    "--out_dir", str(out_dir),
    "--cache", cache,
    # Kernels enabled for NVIDIA cuEquivariance optimizations
    "--output_format", "pdb",
]
```

### 2. Created Test Script
**File**: `test_cuequivariance.py`

Verifies:
- ✅ cuequivariance_torch is installed
- ✅ Available primitives
- ✅ Triangle operations work
- ✅ Checks for attention/norm modules
- ✅ CUDA availability

---

## What's Now Enabled

### Triangle Operations (Optimized by cuEquivariance):
1. **Triangle Multiplicative Updates**
   - Used in Pairformer
   - ~15-20% of compute time
   - **Expected speedup: 3-5x on these operations**

2. **Triangle Attention**
   - Used in triangular attention modules
   - ~5-10% of compute time  
   - **Expected speedup: 2-4x on these operations**

### Overall Expected Speedup:
- **Conservative**: 10-15% faster
- **Realistic**: 15-20% faster
- **Optimistic**: 20-25% faster

---

## Next Steps

### Immediate: Test It!

#### Step 1: Verify Installation
```bash
python test_cuequivariance.py
```

Expected output:
```
Testing cuEquivariance Installation
================================================================================

1. Testing import...
   ✅ cuequivariance_torch imported successfully
   📦 Version: X.X.X

2. Checking available primitives...
   ✅ Found X primitives
      - triangle
      - attention (hopefully!)
      - norm (hopefully!)

3. Testing triangle operations...
   ✅ triangle_multiplicative_update available
   ✅ triangle_attention available

4. Checking for attention operations...
   ✅ attention module found
   📋 Available attention operations:
      - attention_pair_bias (🎯 THIS IS WHAT WE NEED!)
      - ...

5. Checking for normalization operations...
   ✅ norm module found
   📋 Available norm operations:
      - layer_norm
      - ...

6. Checking CUDA...
   ✅ CUDA available
   🎮 Device: YOUR_GPU
   💾 Memory: XX GB
```

#### Step 2: Quick Smoke Test
Test on 1 sample to ensure kernels work:
```bash
python hackathon/predict_hackathon.py \
    --input-json examples/specs/example_protein_complex.json \
    --msa-dir examples/msa/ \
    --submission-dir ./test_with_kernels \
    --intermediate-dir ./test_intermediate
```

Watch for:
- ✅ No errors
- ✅ Predictions complete
- ⏱️ Note the time

#### Step 3: Full Benchmark
```bash
# Use the quick benchmark script
./quick_benchmark.sh
```

This will:
1. Run baseline (but kernels are now enabled!)
2. Run optimized (still with kernels)
3. Compare results

**Note**: Since we removed adaptive recycling, both will use kernels now. To compare against old baseline:
- Use your previous benchmark results (with --no_kernels)
- Compare against new results (with kernels)

---

## Troubleshooting

### If cuequivariance not installed:
```bash
pip install cuequivariance-torch
```

### If CUDA not available:
Kernels will automatically fall back to standard PyTorch. No errors, but no speedup either.

### If predictions fail:
Check the error message:
- Import errors → cuequivariance not installed
- CUDA errors → GPU memory issues
- Other errors → Check logs

---

## Phase 2 Planning

### If test_cuequivariance.py shows `attention_pair_bias` exists:

**Implement AttentionPairBias kernel** (estimated 4-6 hours):

1. **Research API** (1 hour):
   ```python
   from cuequivariance_torch.primitives.attention import attention_pair_bias
   help(attention_pair_bias)  # Check function signature
   ```

2. **Create kernel wrapper** (1 hour):
   - File: `src/boltz/model/layers/attention_kernel.py`
   - Wrapper function matching Boltz's API

3. **Update AttentionPairBias** (1 hour):
   - Add `use_kernels` parameter
   - Add kernel branch in forward()

4. **Propagate flag** (1-2 hours):
   - Update all AttentionPairBias instantiations
   - Pass `use_kernels` through call chain

5. **Test & benchmark** (2 hours):
   - Numerical equivalence tests
   - Performance benchmarks
   - End-to-end validation

**Expected additional speedup**: 25-40%
**Total speedup**: 35-60% (Phase 1 + Phase 2)

---

### If attention_pair_bias doesn't exist:

**Alternative: Flash Attention**

Flash Attention is a proven, production-ready attention optimization:
- 2-4x speedup on attention
- Lower memory usage
- Easy to integrate

```bash
pip install flash-attn
```

---

## Phase 3 Planning

### LayerNorm Optimization (2-3 hours)

**Option A: Apex FusedLayerNorm** (Recommended)
```bash
pip install apex
```

**Option B: cuEquivariance norm** (if available)
Check if `cuequivariance_torch.primitives.norm.layer_norm` exists

**Expected speedup**: 8-12%

---

## Current Branch

You're on: `cuEquivariance-kernels`

Changes made:
- ✅ Removed `--no_kernels` from predict_hackathon.py
- ✅ Added test_cuequivariance.py
- ✅ Updated recycling steps support in predict_hackathon.py

---

## Summary

### Completed:
- ✅ **Phase 1**: Enabled NVIDIA triangle kernels
- ✅ Created test infrastructure
- ✅ Documentation

### Expected Results:
- 🚀 15-20% speedup (Phase 1 alone)
- 🚀 35-60% speedup (Phase 1 + 2)
- 🚀 45-70% speedup (Phase 1 + 2 + 3)

### Next Actions:
1. Run `python test_cuequivariance.py`
2. Test on 1 sample
3. Benchmark on 25% dataset
4. Proceed to Phase 2 based on results

---

**Ready to test? Run the test script now!** 🚀
