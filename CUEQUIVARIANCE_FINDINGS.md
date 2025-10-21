# CRITICAL FINDING: cuEquivariance Already Integrated!

## Discovery

**Boltz already has cuequivariance_torch integrated!**

Found in `src/boltz/model/layers/triangular_mult.py`:
```python
from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update
```

## Current State

### ✅ What's Integrated:
1. **Triangle Multiplicative Updates**
   - `TriangleMultiplicationOutgoing`
   - `TriangleMultiplicationIncoming`
   - Uses `cuequivariance_torch.primitives.triangle.triangle_multiplicative_update`

2. **Kernel Infrastructure**
   - `use_kernels` flag throughout codebase
   - Automatic fallback when kernels unavailable
   - Default: `use_kernels=False`

3. **CLI Support**
   - `--no_kernels` flag to disable (default disables)
   - Need to explicitly NOT use `--no_kernels` to enable

### ❓ What's NOT Using Kernels:
1. **AttentionPairBias** operations
   - In `src/boltz/model/layers/attention.py`
   - In `src/boltz/model/layers/attentionv2.py`
   - **These are the target for optimization!**

2. **Triangular Attention**
   - Has some kernel support in `triangular_attention/primitives.py`
   - But might not be fully optimized

---

## Revised Optimization Strategy

### Option 1: Enable Existing Kernels (EASIEST) ⭐
**Simply test if removing `--no_kernels` flag provides speedup**

**Test**:
```bash
# Current (kernels disabled)
boltz predict input.yaml --no_kernels --out_dir baseline

# With kernels enabled
boltz predict input.yaml --out_dir optimized

# Compare
python benchmark_compare.py baseline optimized
```

**Pros**:
- Zero code changes needed
- Already implemented and tested
- Should work immediately

**Cons**:
- Might not optimize attention operations
- Unknown speedup magnitude

---

### Option 2: Add cuEquivariance Attention (MEDIUM)
**Implement attention using cuequivariance_torch**

**Check if cuequivariance_torch has attention primitives**:
```python
# Research what's available
from cuequivariance_torch import primitives
dir(primitives)  # What's available?
```

**Potential modules to check**:
- `cuequivariance_torch.primitives.attention`
- `cuequivariance_torch.primitives.pair`
- Any attention-related functions

**Implementation**:
If available, follow similar pattern to `triangular_mult.py`:
1. Import cuequivariance attention primitive
2. Create kernel wrapper function
3. Add `if use_kernels:` branch in AttentionPairBias
4. Test equivalence and benchmark

---

### Option 3: Flash Attention Integration (ALTERNATIVE)
**If cuequivariance doesn't have attention, use Flash Attention**

Flash Attention is a well-known, proven attention optimization:
- 2-4x speedup on attention operations
- Lower memory usage
- Easy integration

```bash
pip install flash-attn
```

---

## Recommended Action Plan

### Phase 1: Test Existing Kernels (1 hour) ⭐ START HERE
1. Check if `--no_kernels` is being used in predictions
2. Remove `--no_kernels` flag
3. Benchmark with triangle kernels enabled
4. Measure speedup

**Expected outcome**: 10-30% speedup from triangle operations

---

### Phase 2: Research cuequivariance_torch (2 hours)
1. Install cuequivariance_torch
   ```bash
   pip install cuequivariance-torch
   ```

2. Explore available primitives
   ```python
   import cuequivariance_torch
   from cuequivariance_torch import primitives
   print(dir(primitives))
   ```

3. Check for attention operations
   - Look for `attention`, `pair_attention`, `attention_pair_bias`
   - Check documentation/examples

**Decision point**: 
- ✅ If attention primitives exist → Proceed to Phase 3
- ❌ If not → Pivot to Flash Attention

---

### Phase 3: Implement Attention Kernels (4-6 hours)
**Only if cuequivariance has attention support**

1. Create `kernel_attention_pair_bias()` function
2. Add to `AttentionPairBias.forward()`
3. Test numerical equivalence
4. Benchmark speedup

---

### Phase 4: Benchmark End-to-End (2 hours)
1. Full prediction with kernels enabled
2. Compare quality (RMSD, structures)
3. Compare speed (total time, per-component)
4. Document results

---

## Quick Win: Enable Existing Kernels NOW

Let me check what the hackathon prediction script does:

```bash
grep -r "no_kernels" hackathon/
```

If it uses `--no_kernels`, that's the FIRST thing to change!

---

## Files to Investigate

### 1. Check cuequivariance primitives
```python
# What operations are available?
import cuequivariance_torch.primitives as prims
print(dir(prims))
```

### 2. Check hackathon script
```bash
cat hackathon/predict_hackathon.py | grep "no_kernels"
```

### 3. Read attention implementation
```bash
cat src/boltz/model/layers/attention.py
```

---

## Updated Success Criteria

### Tier 1: Quick Win (Enable existing kernels)
- ✅ Remove `--no_kernels` from predictions
- ✅ 10-30% speedup from triangle operations
- ⏱️ Time: 1 hour

### Tier 2: Attention Optimization (If supported)
- ✅ cuequivariance attention kernels integrated
- ✅ 30-50% speedup on attention
- ⏱️ Time: 1-2 days

### Tier 3: Alternative (Flash Attention)
- ✅ Flash Attention integrated as fallback
- ✅ 2-4x attention speedup
- ⏱️ Time: 1 day

---

## Next Immediate Actions

1. ✅ Create this findings document
2. ⏳ Check if hackathon script uses `--no_kernels`  
3. ⏳ Install/verify cuequivariance-torch
4. ⏳ Explore available primitives
5. ⏳ Create branch and start testing

**Let's start with the easiest win first!**
