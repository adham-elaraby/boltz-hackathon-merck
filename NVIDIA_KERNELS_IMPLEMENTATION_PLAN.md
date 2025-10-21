# 🚀 NVIDIA cuEquivariance Kernels Implementation Plan

## Overview

Implement THREE levels of optimization using NVIDIA's cuEquivariance kernels:
1. **Quick Win** (5 min): Enable existing triangle kernels
2. **Main Win** (4-6 hours): Add NEW AttentionPairBias kernel
3. **Bonus Win** (2-3 hours): Add optimized LayerNorm kernels

**Expected Total Speedup: 40-70%**

---

## Phase 1: Enable Existing Kernels (QUICK WIN) ⚡

### What's Already There:
- ✅ Triangle Multiplicative Updates
- ✅ Triangle Attention  
- ❌ **Currently DISABLED** with `--no_kernels` flag

### Implementation:
**File**: `hackathon/predict_hackathon.py`

**Change** (line ~246):
```python
# REMOVE THIS LINE:
"--no_kernels",
```

### Expected Result:
- **Speedup**: 10-20% (triangle ops are ~15-20% of compute)
- **Time**: 5 minutes
- **Risk**: None (code already tested)

---

## Phase 2: Add AttentionPairBias Kernel (MAIN OPTIMIZATION) 🎯

### Current State:
**File**: `src/boltz/model/layers/attention.py` and `attentionv2.py`

Currently uses standard PyTorch attention:
```python
class AttentionPairBias(nn.Module):
    def forward(self, q, k, v, bias, mask):
        # Standard PyTorch implementation
        # No kernel optimization
```

### New Implementation:

#### Step 1: Research cuEquivariance API
Check what's available in cuequivariance_torch:
```python
# Research script
import cuequivariance_torch
from cuequivariance_torch import primitives

# Check available primitives
print(dir(primitives))

# Look for attention operations
# Expected: primitives.attention.attention_pair_bias or similar
```

#### Step 2: Create Kernel Wrapper
**New File**: `src/boltz/model/layers/attention_kernel.py`

```python
"""Kernel-accelerated attention operations using cuEquivariance."""

import torch
from torch import nn

@torch.compiler.disable
def kernel_attention_pair_bias(
    q,  # Query: [B, H, N, D]
    k,  # Key: [B, H, N, D]
    v,  # Value: [B, H, N, D]
    bias,  # Pair bias: [B, H, N, N]
    mask,  # Attention mask: [B, N, N]
    **kwargs
):
    """
    Optimized AttentionPairBias using cuEquivariance kernels.
    
    This is the NEW NVIDIA kernel for attention with pair bias,
    optimized for AlphaFold-style architectures.
    """
    from cuequivariance_torch.primitives.attention import attention_pair_bias
    
    return attention_pair_bias(
        q=q,
        k=k,
        v=v,
        bias=bias,
        mask=mask,
        **kwargs
    )
```

#### Step 3: Update AttentionPairBias Class
**File**: `src/boltz/model/layers/attention.py`

```python
# Add import at top
from boltz.model.layers.attention_kernel import kernel_attention_pair_bias

class AttentionPairBias(nn.Module):
    """Attention pair bias layer."""
    
    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 8,
        use_kernels: bool = False,  # Add parameter
    ):
        super().__init__()
        self.use_kernels = use_kernels
        # ... rest of init
    
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        bias: Tensor,
        mask: Tensor,
    ) -> Tensor:
        # If kernels enabled, use optimized version
        if self.use_kernels:
            try:
                return kernel_attention_pair_bias(
                    q=q,
                    k=k,
                    v=v,
                    bias=bias,
                    mask=mask,
                )
            except (ImportError, AttributeError) as e:
                # Fallback to standard if kernels unavailable
                print(f"⚠️  Kernel failed, using standard: {e}")
                pass  # Fall through to standard implementation
        
        # Standard PyTorch implementation (existing code)
        # ... existing forward logic
```

#### Step 4: Propagate use_kernels Flag
Update all places that create AttentionPairBias:

**Files to modify**:
- `src/boltz/model/layers/pairformer.py`
- `src/boltz/model/modules/diffusionv2.py`
- Any other files using AttentionPairBias

**Pattern**:
```python
# OLD
self.attention = AttentionPairBias(dim=128, num_heads=8)

# NEW
self.attention = AttentionPairBias(
    dim=128,
    num_heads=8,
    use_kernels=use_kernels  # Propagate flag
)
```

### Expected Result:
- **Speedup**: 30-50% on attention operations
- **Overall**: 25-40% total speedup (attention is ~60% of compute)
- **Time**: 4-6 hours implementation + testing
- **Risk**: Medium (need to match API, verify equivalence)

---

## Phase 3: Add LayerNorm Kernels (BONUS OPTIMIZATION) 🎁

### Current State:
Boltz uses standard PyTorch LayerNorm throughout:
```python
self.norm = nn.LayerNorm(dim)
```

### Optimization Options:

#### Option A: Apex FusedLayerNorm (NVIDIA, Production-Ready)
```bash
pip install apex
```

```python
from apex.normalization import FusedLayerNorm

# Replace nn.LayerNorm with FusedLayerNorm
self.norm = FusedLayerNorm(dim) if use_kernels else nn.LayerNorm(dim)
```

**Pros**:
- ✅ Battle-tested, widely used
- ✅ 2-3x faster than PyTorch LayerNorm
- ✅ Easy installation

**Cons**:
- ❌ Requires CUDA toolkit
- ❌ Another dependency

#### Option B: cuEquivariance LayerNorm (if available)
Check if cuequivariance_torch has LayerNorm primitives:
```python
from cuequivariance_torch.primitives.norm import layer_norm  # Check if exists
```

**Pros**:
- ✅ Same package as other kernels
- ✅ Consistent API

**Cons**:
- ❌ Might not exist yet

#### Option C: torch.compile (PyTorch 2.0+)
Use PyTorch's built-in optimization:
```python
self.norm = torch.compile(nn.LayerNorm(dim))
```

**Pros**:
- ✅ No external dependencies
- ✅ Automatic optimization

**Cons**:
- ❌ Slower than custom kernels
- ❌ Compilation overhead

### Recommended: Apex FusedLayerNorm

**Implementation**:

1. Create wrapper in `src/boltz/model/layers/norm_optimized.py`:
```python
"""Optimized normalization layers."""

import torch
from torch import nn

try:
    from apex.normalization import FusedLayerNorm
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False
    FusedLayerNorm = None


def get_layer_norm(normalized_shape, use_kernels=False, eps=1e-5):
    """
    Get optimized LayerNorm if kernels enabled, otherwise standard.
    
    Args:
        normalized_shape: Shape to normalize over
        use_kernels: Whether to use optimized kernels
        eps: Epsilon for numerical stability
    
    Returns:
        LayerNorm module (optimized or standard)
    """
    if use_kernels and APEX_AVAILABLE:
        return FusedLayerNorm(normalized_shape, eps=eps)
    else:
        return nn.LayerNorm(normalized_shape, eps=eps)
```

2. Replace nn.LayerNorm throughout codebase:
```python
# OLD
from torch import nn
self.norm = nn.LayerNorm(dim)

# NEW
from boltz.model.layers.norm_optimized import get_layer_norm
self.norm = get_layer_norm(dim, use_kernels=self.use_kernels)
```

### Expected Result:
- **Speedup**: 10-15% (LayerNorm is ~10-15% of compute)
- **Time**: 2-3 hours (many files to update)
- **Risk**: Low (Apex is well-tested)

---

## Combined Expected Performance

### Conservative Estimate:
- Triangle kernels: +12%
- AttentionPairBias: +25%
- LayerNorm: +8%
- **Total: ~45% faster**

### Optimistic Estimate:
- Triangle kernels: +18%
- AttentionPairBias: +40%
- LayerNorm: +12%
- **Total: ~70% faster**

---

## Implementation Timeline

| Phase | Task | Time | Cumulative |
|-------|------|------|------------|
| **Phase 1** | Enable existing kernels | 5 min | 5 min |
| | Test & benchmark | 30 min | 35 min |
| **Phase 2** | Research cuEquivariance API | 1 hour | 1.5 hours |
| | Create attention kernel wrapper | 1 hour | 2.5 hours |
| | Update AttentionPairBias | 1 hour | 3.5 hours |
| | Propagate use_kernels flag | 1-2 hours | 5 hours |
| | Test numerical equivalence | 1 hour | 6 hours |
| | Benchmark | 1 hour | 7 hours |
| **Phase 3** | Install Apex | 15 min | 7.25 hours |
| | Create norm wrapper | 30 min | 7.75 hours |
| | Update all LayerNorm calls | 2 hours | 9.75 hours |
| | Test & benchmark | 1 hour | 10.75 hours |
| **Total** | | **~11 hours** | **~1.5 days** |

---

## Testing Strategy

### 1. Numerical Equivalence Tests
**For each optimization**, verify outputs match:

```python
def test_kernel_equivalence():
    # Setup
    q = torch.randn(2, 8, 64, 32).cuda()
    k = torch.randn(2, 8, 64, 32).cuda()
    v = torch.randn(2, 8, 64, 32).cuda()
    bias = torch.randn(2, 8, 64, 64).cuda()
    mask = torch.ones(2, 64, 64).cuda()
    
    # Standard
    attn_std = AttentionPairBias(use_kernels=False).cuda()
    out_std = attn_std(q, k, v, bias, mask)
    
    # Kernel
    attn_kernel = AttentionPairBias(use_kernels=True).cuda()
    out_kernel = attn_kernel(q, k, v, bias, mask)
    
    # Compare
    diff = torch.abs(out_std - out_kernel).max()
    assert diff < 1e-4, f"Outputs differ by {diff}"
    print(f"✅ Numerical equivalence: max diff = {diff:.2e}")
```

### 2. Performance Benchmarks
**Micro-benchmarks** for each component:

```python
def benchmark_attention(use_kernels=False, iterations=100):
    attn = AttentionPairBias(use_kernels=use_kernels).cuda()
    
    # Warmup
    for _ in range(10):
        _ = attn(q, k, v, bias, mask)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = attn(q, k, v, bias, mask)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    return elapsed / iterations
```

### 3. End-to-End Benchmark
Run full prediction pipeline:

```bash
# Standard (no kernels)
time python hackathon/predict_hackathon.py \
    --input-jsonl hackathon_data/datasets/abag_public/abag_public.jsonl \
    --msa-dir hackathon_data/datasets/abag_public/msa/ \
    --submission-dir ./baseline

# With all optimizations
time python hackathon/predict_hackathon.py \
    --input-jsonl hackathon_data/datasets/abag_public/abag_public.jsonl \
    --msa-dir hackathon_data/datasets/abag_public/msa/ \
    --submission-dir ./optimized
```

---

## Phased Rollout Strategy

### Week 1: Phase 1 (Quick Win)
- ✅ Enable existing kernels
- ✅ Verify 10-20% speedup
- ✅ Document and commit

### Week 1-2: Phase 2 (Main Win)
- 🔍 Research cuEquivariance attention API
- 🛠️ Implement kernel wrapper
- 🧪 Test numerical equivalence
- 📊 Benchmark (expect 25-40% additional speedup)
- ✅ Document and commit

### Week 2: Phase 3 (Bonus Win)
- 🛠️ Add LayerNorm optimization
- 🧪 Test and benchmark
- 📊 Final end-to-end benchmarks
- ✅ Document total improvements

---

## Risk Mitigation

### High Priority Risks:

1. **cuEquivariance API mismatch**
   - Risk: AttentionPairBias API might differ from expectations
   - Mitigation: Research thoroughly first, have fallback plan
   - Fallback: Use Flash Attention instead

2. **Numerical differences**
   - Risk: Kernel outputs might differ slightly
   - Mitigation: Extensive testing with tolerance checks
   - Acceptable: <1e-4 difference

3. **Installation issues**
   - Risk: cuequivariance/apex might not install easily
   - Mitigation: Document installation, provide alternatives
   - Fallback: Make optimizations optional

### Medium Priority Risks:

4. **Performance not as expected**
   - Risk: Speedup might be less than estimated
   - Mitigation: Benchmark each phase separately
   - Decision: Keep if >10% improvement

5. **Code maintenance**
   - Risk: More complex codebase
   - Mitigation: Clear separation, good documentation
   - Keep fallbacks for easy debugging

---

## Success Criteria

### Must Have (Phase 1):
- ✅ Triangle kernels enabled
- ✅ >10% speedup
- ✅ No breaking changes

### Should Have (Phase 2):
- ✅ AttentionPairBias kernel integrated
- ✅ >30% total speedup
- ✅ Numerical equivalence verified

### Nice to Have (Phase 3):
- ✅ LayerNorm optimized
- ✅ >45% total speedup
- ✅ Clean, maintainable code

---

## Next Steps

1. **Immediate**: Remove `--no_kernels` line (5 min)
2. **Today**: Test Phase 1, verify speedup (30 min)
3. **This Week**: Research and implement Phase 2 (6-7 hours)
4. **Next Week**: Add Phase 3 if time permits (3 hours)

**Ready to start with Phase 1?**
