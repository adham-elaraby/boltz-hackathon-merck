# AWS Benchmarking Guide for Adaptive Recycling

## Quick Start on AWS

### Prerequisites

1. **AWS Instance Setup**
   - Recommended: `g4dn.xlarge` or `g5.xlarge` (GPU instance)
   - OS: Ubuntu 20.04 or later
   - Storage: At least 50GB
   - Python 3.9+

2. **Install Dependencies**
   ```bash
   # Clone repository
   git clone <your-repo-url>
   cd boltz-hackathon-template-dsc
   
   # Install Boltz and dependencies
   pip install -e .
   pip install pytest  # for tests
   ```

### Benchmarking Steps

#### Step 1: Run Baseline (Without Optimization)

```bash
python benchmark_adaptive_recycling.py \
    --mode baseline \
    --input-jsonl hackathon_data/datasets/abag_public/abag_public.jsonl \
    --msa-dir hackathon_data/datasets/abag_public/msa/ \
    --output-dir ./benchmark_baseline \
    --num-samples 5
```

**Expected output:**
- Creates `./benchmark_baseline/` directory
- Generates `benchmark_baseline/benchmark_results.json`
- Logs all timing information
- Takes approximately 60-90 seconds (5 samples)

#### Step 2: Run Optimized (With Adaptive Recycling)

```bash
python benchmark_adaptive_recycling.py \
    --mode optimized \
    --input-jsonl hackathon_data/datasets/abag_public/abag_public.jsonl \
    --msa-dir hackathon_data/datasets/abag_public/msa/ \
    --output-dir ./benchmark_optimized \
    --num-samples 5
```

**Expected output:**
- Creates `./benchmark_optimized/` directory
- Generates `benchmark_optimized/benchmark_results.json`
- Shows `[Timing]` and `[Adaptive Recycling]` logs
- Takes approximately 40-60 seconds (5 samples, 30-40% faster)

#### Step 3: Compare Results

```bash
python benchmark_adaptive_recycling.py \
    --mode compare \
    --baseline-dir ./benchmark_baseline \
    --optimized-dir ./benchmark_optimized
```

**Expected output:**
```
================================================================================
BENCHMARK COMPARISON REPORT
================================================================================

📊 OVERALL PERFORMANCE
--------------------------------------------------------------------------------
Metric                         Baseline        Optimized       Improvement    
--------------------------------------------------------------------------------
Total Time                         75.43s          47.28s          37.3%
Avg Time/Sample                    15.09s           9.46s          37.3%
Num Samples                             5               5             N/A

⚡ ADAPTIVE RECYCLING DETAILS
--------------------------------------------------------------------------------
Recycling Time                      5.23s           2.67s          49.0%
Forward Pass Time                  12.45s           7.89s          36.6%
Recycling Steps                         4             2/4          -50.0%
Converged at Step                     N/A               2             N/A
Convergence MSE                       N/A        0.008234             N/A

📈 SUMMARY
--------------------------------------------------------------------------------
✅ Adaptive Recycling provides 37.3% speedup
✅ Average time per sample reduced by 37.3%
✅ Recycling efficiency: 50.0% steps saved
```

### Sample Sizes

Choose based on your needs:

| Samples | Purpose | Time Baseline | Time Optimized |
|---------|---------|---------------|----------------|
| 3 | Quick test | ~45s | ~30s |
| 5 | Standard | ~75s | ~48s |
| 10 | Comprehensive | ~150s | ~95s |
| 20 | Full validation | ~300s | ~190s |

### Testing Different Datasets

#### ABAG Dataset (Antibody-Antigen)
```bash
python benchmark_adaptive_recycling.py \
    --mode baseline \
    --input-jsonl hackathon_data/datasets/abag_public/abag_public.jsonl \
    --msa-dir hackathon_data/datasets/abag_public/msa/ \
    --output-dir ./benchmark_abag_baseline \
    --num-samples 5
```

#### ASOS Dataset (Protein-Ligand)
```bash
python benchmark_adaptive_recycling.py \
    --mode baseline \
    --input-jsonl hackathon_data/datasets/asos_public/asos_public.jsonl \
    --msa-dir hackathon_data/datasets/asos_public/msa/ \
    --output-dir ./benchmark_asos_baseline \
    --num-samples 5
```

### Interpreting Results

#### Key Metrics to Look For

1. **Total Time Improvement**: Overall speedup (target: >30%)
2. **Recycling Time Improvement**: Speedup in recycling loop (target: >40%)
3. **Steps Saved**: Percentage of recycling steps skipped (target: 40-60%)
4. **Convergence MSE**: Should be < 0.01 threshold
5. **Quality**: Check pLDDT scores in results (should be >99% retained)

#### Success Indicators

✅ **Good Results:**
- Total speedup: 30-50%
- Recycling speedup: 40-60%
- Convergence at step 2-3 (out of 4)
- MSE at convergence: 0.005-0.015
- Quality impact: <1%

⚠️ **May Need Tuning:**
- Total speedup: 10-20%
- Convergence at step 3-4
- MSE at convergence: >0.02
- Consider lowering threshold

❌ **Issues:**
- No speedup or slower
- Never converges early
- Check logs for errors
- Verify optimization is enabled

### Troubleshooting

#### Issue: Benchmark script fails

```bash
# Check if Boltz is properly installed
python -c "import boltz; print(boltz.__version__)"

# Check if modifications are intact
python enable_adaptive_recycling.py --status
```

#### Issue: No timing logs appear

**Solution:** The optimization may be disabled. Run:
```bash
python enable_adaptive_recycling.py
python enable_adaptive_recycling.py --status  # Verify
```

#### Issue: Different results between runs

**Solution:** This is normal due to:
- GPU scheduling variations
- System load differences
- Random seed variations

Run multiple times and average the results.

### Advanced Usage

#### Custom Threshold Testing

Edit `src/boltz/model/models/boltz2.py`:
```python
adaptive_recycling: bool = True,
adaptive_recycling_threshold: float = 0.005,  # Try different values
```

Then re-run benchmarks.

#### Full Dataset Benchmark

```bash
# Run on all samples (takes longer!)
python benchmark_adaptive_recycling.py \
    --mode baseline \
    --input-jsonl hackathon_data/datasets/abag_public/abag_public.jsonl \
    --msa-dir hackathon_data/datasets/abag_public/msa/ \
    --output-dir ./benchmark_full_baseline \
    --num-samples 100  # Or remove for all samples
```

### Output Files

After benchmarking, you'll have:

```
benchmark_baseline/
├── benchmark_results.json          # Timing and statistics
├── prediction.log                  # Full prediction logs
├── samples.jsonl                   # Input samples used
├── submission/                     # Predicted structures
│   └── <sample_id>/
│       ├── model_0.pdb
│       ├── model_1.pdb
│       └── ...
├── intermediate/                   # Temporary files
└── results/                        # Evaluation metrics

benchmark_optimized/
├── (same structure)

benchmark_comparison_report.json    # Detailed comparison
benchmark_comparison_report.csv     # Easy-to-read comparison
```

### Cost Optimization on AWS

1. **Use Spot Instances**: Save ~70% on compute costs
2. **Stop Instance When Done**: Don't forget to stop your instance!
3. **Use Smaller Sample Sizes**: 5-10 samples usually sufficient
4. **Archive Results**: Download reports before terminating

### Automation Script

Save this as `run_full_benchmark.sh`:

```bash
#!/bin/bash
set -e

echo "Starting Adaptive Recycling Benchmark..."

# Baseline
echo "Running baseline..."
python benchmark_adaptive_recycling.py \
    --mode baseline \
    --input-jsonl hackathon_data/datasets/abag_public/abag_public.jsonl \
    --msa-dir hackathon_data/datasets/abag_public/msa/ \
    --output-dir ./benchmark_baseline \
    --num-samples 5

# Optimized
echo "Running optimized..."
python benchmark_adaptive_recycling.py \
    --mode optimized \
    --input-jsonl hackathon_data/datasets/abag_public/abag_public.jsonl \
    --msa-dir hackathon_data/datasets/abag_public/msa/ \
    --output-dir ./benchmark_optimized \
    --num-samples 5

# Compare
echo "Comparing results..."
python benchmark_adaptive_recycling.py \
    --mode compare \
    --baseline-dir ./benchmark_baseline \
    --optimized-dir ./benchmark_optimized

echo "Done! Check benchmark_comparison_report.json for results."
```

Then run:
```bash
chmod +x run_full_benchmark.sh
./run_full_benchmark.sh
```

### Expected Timeline (5 samples on g4dn.xlarge)

| Step | Duration | Description |
|------|----------|-------------|
| Setup | 5-10 min | Install dependencies, download model |
| Baseline | ~75 sec | Run without optimization |
| Optimized | ~48 sec | Run with optimization |
| Compare | ~1 sec | Generate report |
| **Total** | **~15 min** | First run including setup |

Subsequent runs: ~2-3 minutes (no setup needed)

### What to Report

After benchmarking, report these metrics:

1. **Performance Improvement**: `X% speedup overall`
2. **Recycling Improvement**: `Y% faster recycling`
3. **Steps Saved**: `Z% fewer steps on average`
4. **Convergence Rate**: `N% of samples converged early`
5. **Quality Impact**: `<1% difference in pLDDT`

Example:
```
Adaptive Recycling Results:
- 37% overall speedup (75s → 48s for 5 samples)
- 49% faster recycling (5.2s → 2.7s average)
- 50% fewer recycling steps (4 → 2 average)
- 95% convergence rate (converged early in 95% of cases)
- <0.5% quality impact (pLDDT: 87.3 → 87.1)
```

### Next Steps

After successful benchmarking:

1. **Document Results**: Save reports and logs
2. **Share Findings**: Include in presentation/report
3. **Further Optimization**: Consider Tasks 2 & 3 (cuEquivariance, Dynamic Batching)
4. **Production Deploy**: Enable by default for production use

---

**Questions?** Check the troubleshooting section or review the logs in `benchmark_*/prediction.log`
