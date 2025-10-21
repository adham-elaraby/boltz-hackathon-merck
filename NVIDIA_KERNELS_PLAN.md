# NVIDIA cuEquivariance Kernels Optimization Plan

## Executive Summary

Replace Boltz's standard AttentionPairBias operations with NVIDIA's cuEquivariance optimized kernels for significant performance improvements on GPU.

---

## Current State Analysis

### ✅ What We Know:
1. **Boltz already has kernel infrastructure**
   - `use_kernels` flag exists throughout codebase
   - Default: `use_kernels=False`
   - CLI flag: `--no_kernels` (disables kernels)
   
2. **AttentionPairBias is a key operation**
   - Found in: `src/boltz/model/layers/attention.py`
   - Also: `src/boltz/model/layers/attentionv2.py`
   - Used extensively in Pairformer and diffusion modules

3. **Attention is computationally expensive**
   - Runs multiple times during recycling (3-6 steps)
   - Runs many times during diffusion (200 sampling steps)
   - Major performance bottleneck

### ❓ What We Need to Investigate:
1. What kernels does Boltz currently use (if any)?
2. What is cuEquivariance and how does it work?
3. Is it compatible with Boltz's attention operations?
4. What are the installation requirements?

---

## Research Phase

### Step 1: Understand Current Kernel Usage
**Goal**: Find out what `use_kernels=True` currently does in Boltz

**Actions**:
- [ ] Search for kernel implementations in `src/boltz/model/layers/`
- [ ] Check if there are existing optimized kernels
- [ ] Understand the fallback mechanism

**Commands**:
```bash
# Search for kernel-related files
find src/boltz -name "*kernel*" -o -name "*cuda*"

# Check what use_kernels controls
grep -r "if use_kernels" src/boltz/
```

### Step 2: Research cuEquivariance
**Goal**: Understand NVIDIA's cuEquivariance library

**Actions**:
- [ ] Use Context7 MCP to get cuequivariance-torch documentation
- [ ] Understand what operations it optimizes
- [ ] Check compatibility with PyTorch version
- [ ] Review installation requirements

**Questions to Answer**:
- Does it optimize attention operations?
- What is the API structure?
- What are the dependencies?
- Is it production-ready?

### Step 3: Analyze AttentionPairBias
**Goal**: Understand the current implementation

**Actions**:
- [ ] Read `src/boltz/model/layers/attention.py`
- [ ] Identify the computationally expensive parts
- [ ] Map operations to potential cuEquivariance equivalents

---

## Implementation Plan

### Phase 1: Setup & Environment (Day 1)

#### 1.1 Create New Branch
```bash
git checkout -b nvidia-cuequivariance-kernels
git push -u origin nvidia-cuequivariance-kernels
```

#### 1.2 Install cuequivariance-torch
```bash
# Research installation first with Context7
pip install cuequivariance-torch
```

#### 1.3 Verify Installation
Create `tests/test_cuequivariance_install.py`:
```python
def test_import():
    try:
        import cuequivariance
        print(f"✅ cuEquivariance version: {cuequivariance.__version__}")
        return True
    except ImportError:
        print("❌ cuEquivariance not installed")
        return False
```

---

### Phase 2: Wrapper Implementation (Day 1-2)

#### 2.1 Create Optimized Attention Layer
**File**: `src/boltz/model/layers/attention_optimized.py`

**Structure**:
```python
"""
Optimized attention layer using NVIDIA cuEquivariance kernels.
Falls back to standard implementation if kernels unavailable.
"""

import torch
from torch import nn
try:
    import cuequivariance
    CUEQUIVARIANCE_AVAILABLE = True
except ImportError:
    CUEQUIVARIANCE_AVAILABLE = False

from .attention import AttentionPairBias as StandardAttentionPairBias


class OptimizedAttentionPairBias(nn.Module):
    """
    Drop-in replacement for AttentionPairBias using cuEquivariance kernels.
    
    Automatically falls back to standard implementation if:
    - cuEquivariance not installed
    - use_kernels=False
    - CUDA not available
    """
    
    def __init__(self, *args, use_kernels=False, **kwargs):
        super().__init__()
        
        # Decide which implementation to use
        self.use_optimized = (
            use_kernels and 
            CUEQUIVARIANCE_AVAILABLE and 
            torch.cuda.is_available()
        )
        
        if self.use_optimized:
            # Initialize cuEquivariance version
            self._init_cuequivariance(*args, **kwargs)
        else:
            # Fall back to standard
            self.standard = StandardAttentionPairBias(*args, **kwargs)
    
    def _init_cuequivariance(self, *args, **kwargs):
        """Initialize cuEquivariance optimized kernels."""
        # TODO: Implement based on cuEquivariance API
        pass
    
    def forward(self, *args, **kwargs):
        """Forward pass with automatic fallback."""
        if self.use_optimized:
            return self._forward_optimized(*args, **kwargs)
        else:
            return self.standard(*args, **kwargs)
    
    def _forward_optimized(self, *args, **kwargs):
        """Optimized forward using cuEquivariance."""
        # TODO: Implement
        pass
```

#### 2.2 Integration Points
**Files to modify**:
- `src/boltz/model/layers/__init__.py` - Export optimized version
- Optionally add config flag to toggle optimization

---

### Phase 3: Testing & Validation (Day 2)

#### 3.1 Numerical Equivalence Test
**File**: `tests/test_attention_equivalence.py`

```python
"""Test that optimized attention matches standard implementation."""

import torch
import pytest
from boltz.model.layers.attention import AttentionPairBias
from boltz.model.layers.attention_optimized import OptimizedAttentionPairBias


def test_numerical_equivalence():
    """Verify outputs match within tolerance."""
    # Setup
    batch_size, seq_len, dim = 2, 64, 128
    
    # Create inputs
    q = torch.randn(batch_size, seq_len, dim).cuda()
    k = torch.randn(batch_size, seq_len, dim).cuda()
    v = torch.randn(batch_size, seq_len, dim).cuda()
    bias = torch.randn(batch_size, seq_len, seq_len).cuda()
    
    # Standard version
    standard = AttentionPairBias(...).cuda()
    out_standard = standard(q, k, v, bias)
    
    # Optimized version
    optimized = OptimizedAttentionPairBias(..., use_kernels=True).cuda()
    out_optimized = optimized(q, k, v, bias)
    
    # Compare
    diff = torch.abs(out_standard - out_optimized).max()
    assert diff < 1e-5, f"Outputs differ by {diff}"
```

#### 3.2 Performance Benchmark
**File**: `benchmark_attention_kernels.py`

```python
"""Benchmark attention performance: standard vs optimized."""

import time
import torch
from boltz.model.layers.attention import AttentionPairBias
from boltz.model.layers.attention_optimized import OptimizedAttentionPairBias


def benchmark_attention(seq_len=128, num_iterations=100):
    """Benchmark attention operations."""
    # Setup
    q = torch.randn(1, seq_len, 256).cuda()
    k = torch.randn(1, seq_len, 256).cuda()
    v = torch.randn(1, seq_len, 256).cuda()
    bias = torch.randn(1, seq_len, seq_len).cuda()
    
    # Standard
    standard = AttentionPairBias(...).cuda()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        _ = standard(q, k, v, bias)
    torch.cuda.synchronize()
    time_standard = time.time() - start
    
    # Optimized
    optimized = OptimizedAttentionPairBias(..., use_kernels=True).cuda()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        _ = optimized(q, k, v, bias)
    torch.cuda.synchronize()
    time_optimized = time.time() - start
    
    # Report
    speedup = time_standard / time_optimized
    print(f"Standard:  {time_standard:.3f}s")
    print(f"Optimized: {time_optimized:.3f}s")
    print(f"Speedup:   {speedup:.2f}x")
    
    return speedup
```

---

### Phase 4: Integration & End-to-End Testing (Day 3)

#### 4.1 Update Model to Use Optimized Attention
**Approach**: Minimal changes, use factory pattern

Option A: Direct replacement in imports
```python
# In files that import AttentionPairBias
from boltz.model.layers.attention_optimized import OptimizedAttentionPairBias as AttentionPairBias
```

Option B: Configuration-based selection
```python
# In model initialization
if use_optimized_kernels:
    from boltz.model.layers.attention_optimized import OptimizedAttentionPairBias as Attention
else:
    from boltz.model.layers.attention import AttentionPairBias as Attention
```

#### 4.2 End-to-End Prediction Test
Run full prediction with optimized kernels:
```bash
# Standard
boltz predict input.yaml --no_kernels --out_dir output_standard

# Optimized
boltz predict input.yaml --out_dir output_optimized

# Compare outputs
python compare_predictions.py output_standard output_optimized
```

---

## Success Criteria

### ✅ Must Have:
1. **Numerical Equivalence**: Output difference < 1e-5
2. **Performance Improvement**: At least 1.5x speedup on attention operations
3. **Graceful Fallback**: Works without cuEquivariance installed
4. **No Breaking Changes**: Existing code still works
5. **GPU Only**: Only activates on CUDA devices

### 🎯 Nice to Have:
1. **2x+ Speedup**: Significant performance gain
2. **Memory Efficiency**: Lower GPU memory usage
3. **Easy Toggle**: Simple flag to enable/disable
4. **Documentation**: Clear usage instructions

---

## Risk Assessment

### High Risk:
- **cuEquivariance API mismatch**: Operations might not map directly
  - **Mitigation**: Research API thoroughly first
  
- **Numerical instability**: Different precision handling
  - **Mitigation**: Extensive testing with tolerance checks

### Medium Risk:
- **Installation complexity**: Dependencies might conflict
  - **Mitigation**: Document installation clearly, use fallback

- **Limited speedup**: Might not be worth the complexity
  - **Mitigation**: Benchmark early, pivot if needed

### Low Risk:
- **Code maintenance**: Additional code to maintain
  - **Mitigation**: Keep wrapper simple, well-documented

---

## Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| **Research** | 2-3 hours | Understand current code, cuEquivariance API |
| **Setup** | 1 hour | Branch, install, verify |
| **Implementation** | 4-6 hours | Write wrapper, integrate |
| **Testing** | 2-3 hours | Equivalence, performance tests |
| **Integration** | 2-3 hours | End-to-end testing, docs |
| **Total** | **1-2 days** | |

---

## Next Steps

### Immediate Actions:
1. ✅ Create this plan document
2. ⏳ Research current kernel usage in Boltz
3. ⏳ Use Context7 to understand cuequivariance-torch
4. ⏳ Create new branch
5. ⏳ Start implementation

### Questions to Answer First:
1. What does `use_kernels=True` currently do in Boltz?
2. What operations does cuEquivariance optimize?
3. Is cuEquivariance compatible with Boltz's PyTorch version?
4. What is the expected speedup?

---

## Decision Points

### Go/No-Go Criteria:
After research phase, decide whether to proceed based on:
- ✅ cuEquivariance supports required operations
- ✅ Installation is straightforward
- ✅ Expected speedup > 1.5x
- ✅ Integration effort is reasonable

If any criteria fail, pivot to alternative optimization (e.g., Flash Attention, memory optimization).

---

## Notes
- Keep changes minimal and focused
- Maintain backward compatibility
- Document everything
- Benchmark early and often
