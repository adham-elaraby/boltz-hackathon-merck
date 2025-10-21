# 🎯 QUICK WIN: Enable cuEquivariance Kernels

## THE PROBLEM

**The hackathon prediction script is DISABLING the optimized NVIDIA kernels!**

Found in `hackathon/predict_hackathon.py` line 246:
```python
fixed = [
    "boltz", "predict", str(yaml_path),
    "--devices", "1",
    "--out_dir", str(out_dir),
    "--cache", cache,
    "--no_kernels",  # ← THIS IS KILLING PERFORMANCE!
    "--output_format", "pdb",
]
```

## THE SOLUTION

**Simply remove `"--no_kernels"` from the command!**

This will enable:
- Triangle Multiplicative Updates using cuEquivariance kernels
- Potential other optimized operations
- **Expected speedup: 10-40%** with ZERO code complexity

---

## Implementation Plan

### Step 1: Create New Branch
```bash
git checkout -b enable-nvidia-kernels
```

### Step 2: Remove --no_kernels Flag
**File**: `hackathon/predict_hackathon.py`

**Change**:
```python
# BEFORE (line ~240-248)
fixed = [
    "boltz", "predict", str(yaml_path),
    "--devices", "1",
    "--out_dir", str(out_dir),
    "--cache", cache,
    "--no_kernels",  # ← REMOVE THIS LINE
    "--output_format", "pdb",
]

# AFTER
fixed = [
    "boltz", "predict", str(yaml_path),
    "--devices", "1",
    "--out_dir", str(out_dir),
    "--cache", cache,
    "--output_format", "pdb",
]
```

### Step 3: Test Installation
Verify cuequivariance-torch is installed:
```bash
python -c "import cuequivariance_torch; print(cuequivariance_torch.__version__)"
```

If not installed:
```bash
pip install cuequivariance-torch
```

### Step 4: Run Benchmark
Use existing benchmark infrastructure:

```bash
# Baseline (with --no_kernels, current state)
# Already have results from previous run

# Optimized (without --no_kernels, kernels enabled)
python hackathon/predict_hackathon.py \
    --input-jsonl hackathon_data/datasets/abag_public/abag_public.jsonl \
    --msa-dir hackathon_data/datasets/abag_public/msa/ \
    --submission-dir ./submission_with_kernels \
    --intermediate-dir ./intermediate_with_kernels

# Time and compare
```

---

## Expected Results

### Best Case: 30-40% Speedup
- Triangle operations are heavily used in Pairformer
- cuEquivariance kernels are highly optimized
- Significant portion of compute time

### Realistic Case: 15-25% Speedup
- Triangle ops are fraction of total time
- Still have unoptimized attention
- But meaningful improvement

### Worst Case: 5-10% Speedup
- Triangle ops are small fraction
- Overhead from kernel calls
- Still worth it for free optimization

---

## What This Enables

### Currently Optimized (with kernels):
✅ Triangle Multiplicative Updates (Outgoing)
✅ Triangle Multiplicative Updates (Incoming)  
✅ Possibly triangle attention primitives

### Still Not Optimized:
❌ AttentionPairBias operations
❌ MSA attention
❌ Diffusion attention

**Next phase**: Optimize attention operations too!

---

## Risk Assessment

### Extremely Low Risk:
- ✅ Code already exists and is tested
- ✅ Boltz team clearly intended this to be used
- ✅ Simple flag removal, easy to revert
- ✅ Graceful fallback if kernels unavailable
- ✅ No API changes needed

### Why was --no_kernels added?
Possible reasons:
1. Compatibility: Some systems might not have CUDA/kernels
2. Debugging: Easier to debug without kernels
3. Oversight: Might have been temporary for hackathon simplicity

**For production/benchmarking**: Should definitely use kernels!

---

## Timeline

| Task | Time | Status |
|------|------|--------|
| Create branch | 1 min | ⏳ |
| Remove --no_kernels | 1 min | ⏳ |
| Verify cuequivariance | 5 min | ⏳ |
| Test single prediction | 5 min | ⏳ |
| Run benchmark (25%) | 15-20 min | ⏳ |
| Analyze results | 10 min | ⏳ |
| **TOTAL** | **~40 minutes** | |

---

## Benchmarking Strategy

### Quick Test (5 minutes):
```bash
# Run on 1 sample to verify it works
time python hackathon/predict_hackathon.py \
    --input-json examples/specs/example_protein_complex.json \
    --msa-dir examples/msa/ \
    --submission-dir ./test_kernels
```

### Full Benchmark (20 minutes):
```bash
# Use 25% of dataset (2-3 samples)
# Compare against previous baseline results
./quick_benchmark.sh  # After removing --no_kernels
```

### What to Compare:
1. **Total runtime**: Overall speedup
2. **Per-component timing**: Which parts got faster
3. **Structure quality**: Verify outputs are equivalent
4. **GPU memory**: Check if memory usage changed

---

## After This Quick Win

### If 20%+ speedup → Great!
1. ✅ Commit and document
2. ⏭️ Move to Phase 2: Optimize attention
3. 📊 Potential for even more gains

### If 10-20% speedup → Good!
1. ✅ Keep the change (free speedup)
2. 🤔 Analyze where time is spent
3. ⏭️ Focus on biggest bottlenecks

### If <10% speedup → Still worth it
1. ✅ Keep the change (no downside)
2. 🔍 Profile to find main bottlenecks
3. ⏭️ Consider other optimizations

---

## Commands to Run NOW

### 1. Create branch and make change
```bash
# Create branch
git checkout -b enable-nvidia-kernels

# Edit the file (remove --no_kernels line)
code hackathon/predict_hackathon.py

# Verify cuequivariance is installed
python -c "import cuequivariance_torch; print('✅ cuEquivariance available')"
```

### 2. Quick smoke test
```bash
# Test on 1 sample to ensure it works
python hackathon/predict_hackathon.py \
    --input-json examples/specs/example_protein_complex.json \
    --msa-dir examples/msa/ \
    --submission-dir ./test_with_kernels \
    --intermediate-dir ./test_intermediate
```

### 3. If successful, run full benchmark
```bash
# Modify quick_benchmark.sh to use new code
# Run benchmark
./quick_benchmark.sh
```

---

## Success! 🎉

This is the EASIEST optimization possible:
- ✅ 1 line change
- ✅ ~40 minutes total time
- ✅ 10-40% expected speedup
- ✅ Zero added complexity
- ✅ Production-ready code

**Let's do this!**
