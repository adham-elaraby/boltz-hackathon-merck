# Pre-Benchmark Checklist ✅

Use this checklist before running benchmarks on AWS to ensure everything is ready.

## ☑️ Code Changes

- [x] **Modified `src/boltz/model/models/boltz2.py`**
  - [x] Added `import time`
  - [x] Added `adaptive_recycling` parameter
  - [x] Added `adaptive_recycling_threshold` parameter
  - [x] Implemented convergence detection
  - [x] Added timing instrumentation
  - [x] Added logging

- [x] **Created test suite**
  - [x] `tests/test_adaptive_recycling.py` with 20+ tests

- [x] **Created benchmark script**
  - [x] `benchmark_adaptive_recycling.py` for automated testing

- [x] **Created helper scripts**
  - [x] `enable_adaptive_recycling.py` for easy enable/disable
  - [x] `quick_benchmark.sh` for automated benchmarking

- [x] **Created documentation**
  - [x] ADAPTIVE_RECYCLING_README.md
  - [x] BEFORE_AFTER_COMPARISON.md
  - [x] OPTIMIZATION_SUMMARY.md
  - [x] AWS_BENCHMARKING_GUIDE.md
  - [x] READY_TO_BENCHMARK.md
  - [x] VISUAL_GUIDE.md

## ☑️ Pre-AWS Checks

Run these locally before going to AWS:

### 1. Syntax Check
```bash
python -c "from boltz.model.models.boltz2 import Boltz2; print('✅ OK')"
```
**Expected:** `✅ OK`

### 2. Test Suite
```bash
pytest tests/test_adaptive_recycling.py -v --tb=short
```
**Expected:** All tests pass

### 3. Scripts Exist
```bash
ls benchmark_adaptive_recycling.py enable_adaptive_recycling.py quick_benchmark.sh
```
**Expected:** All three files found

### 4. Documentation Complete
```bash
ls ADAPTIVE_RECYCLING_README.md READY_TO_BENCHMARK.md AWS_BENCHMARKING_GUIDE.md
```
**Expected:** All docs found

### 5. Enable/Disable Works
```bash
python enable_adaptive_recycling.py --status
python enable_adaptive_recycling.py
python enable_adaptive_recycling.py --status
python enable_adaptive_recycling.py --disable
```
**Expected:** Status changes correctly

## ☑️ AWS Setup Checks

Once on AWS instance:

### 1. Python Version
```bash
python --version
```
**Expected:** Python 3.9 or higher

### 2. GPU Available
```bash
nvidia-smi
```
**Expected:** Shows GPU info (T4, A10G, etc.)

### 3. Install Dependencies
```bash
pip install -e .
pip install pytest
```
**Expected:** No errors

### 4. Dataset Available
```bash
ls hackathon_data/datasets/abag_public/abag_public.jsonl
ls -d hackathon_data/datasets/abag_public/msa/
```
**Expected:** Files/directories exist

### 5. Test Import
```bash
python -c "import boltz; from boltz.model.models.boltz2 import Boltz2; print('✅ Imports OK')"
```
**Expected:** `✅ Imports OK`

### 6. Optimization Status
```bash
python enable_adaptive_recycling.py --status
```
**Expected:** Shows current status

## ☑️ Pre-Benchmark Verification

Before running the full benchmark:

### Quick Test (30 seconds)
```bash
# Create 1-sample test
head -n 1 hackathon_data/datasets/abag_public/abag_public.jsonl > test.jsonl

# Test baseline
python benchmark_adaptive_recycling.py \
    --mode baseline \
    --input-jsonl test.jsonl \
    --msa-dir hackathon_data/datasets/abag_public/msa/ \
    --output-dir ./test_baseline \
    --num-samples 1

# Check output
ls test_baseline/benchmark_results.json
cat test_baseline/benchmark_results.json
```

**Expected:**
- No errors
- JSON file created
- Contains timing data

## ☑️ Ready to Run Full Benchmark

If all checks pass above, you're ready to run:

```bash
./quick_benchmark.sh
```

Or manually:

```bash
# Baseline
python benchmark_adaptive_recycling.py \
    --mode baseline \
    --input-jsonl hackathon_data/datasets/abag_public/abag_public.jsonl \
    --msa-dir hackathon_data/datasets/abag_public/msa/ \
    --output-dir ./benchmark_baseline \
    --num-samples 5

# Optimized
python benchmark_adaptive_recycling.py \
    --mode optimized \
    --input-jsonl hackathon_data/datasets/abag_public/abag_public.jsonl \
    --msa-dir hackathon_data/datasets/abag_public/msa/ \
    --output-dir ./benchmark_optimized \
    --num-samples 5

# Compare
python benchmark_adaptive_recycling.py \
    --mode compare \
    --baseline-dir ./benchmark_baseline \
    --optimized-dir ./benchmark_optimized
```

## ☑️ Post-Benchmark Checklist

After benchmark completes:

- [ ] **Check baseline results**
  - [ ] `benchmark_baseline/benchmark_results.json` exists
  - [ ] `benchmark_baseline/prediction.log` has no errors
  - [ ] Structures generated in `benchmark_baseline/submission/`

- [ ] **Check optimized results**
  - [ ] `benchmark_optimized/benchmark_results.json` exists
  - [ ] `benchmark_optimized/prediction.log` has `[Timing]` and `[Adaptive Recycling]` logs
  - [ ] Structures generated in `benchmark_optimized/submission/`

- [ ] **Check comparison report**
  - [ ] `benchmark_comparison_report.json` exists
  - [ ] `benchmark_comparison_report.csv` exists
  - [ ] Shows speedup > 20%

- [ ] **Verify quality**
  - [ ] Check pLDDT scores in results
  - [ ] Compare structures visually (optional)
  - [ ] Quality difference < 1%

## ☑️ Documentation Checklist

Before final submission:

- [ ] Save all reports (JSON, CSV)
- [ ] Save example logs showing timing improvements
- [ ] Take screenshots of comparison output
- [ ] Document hardware specs (GPU model, instance type)
- [ ] Document dataset and sample size
- [ ] Calculate and document:
  - [ ] Overall speedup percentage
  - [ ] Recycling speedup percentage
  - [ ] Steps saved percentage
  - [ ] Convergence rate
  - [ ] Quality impact

## ☑️ Troubleshooting Reference

### Common Issues

| Issue | Solution | Check Command |
|-------|----------|---------------|
| Import error | `pip install -e .` | `python -c "import boltz"` |
| No timing logs | Enable optimization | `python enable_adaptive_recycling.py` |
| Dataset missing | Check path | `ls hackathon_data/datasets/` |
| Script not found | Wrong directory | `pwd; ls *.py` |
| Tests fail | Check modifications | `pytest tests/test_adaptive_recycling.py` |

### Debug Commands

```bash
# Check what's installed
pip list | grep boltz

# Verify file modifications
grep "adaptive_recycling" src/boltz/model/models/boltz2.py | head -n 3

# Check logs
tail -n 50 benchmark_optimized/prediction.log

# Verify GPU usage
watch -n 1 nvidia-smi
```

## ☑️ Final Checks

Before considering the task complete:

- [ ] **Performance**: Achieved 30-50% speedup
- [ ] **Quality**: <1% impact on predictions
- [ ] **Tests**: All unit tests passing
- [ ] **Documentation**: Complete and clear
- [ ] **Reproducible**: Can be run by others
- [ ] **Reports**: Generated and saved
- [ ] **Code quality**: No errors, clean implementation

## ✅ Ready to Deploy

If all checks pass:

- ✅ Code is production-ready
- ✅ Optimization provides significant benefit
- ✅ Quality is maintained
- ✅ Well-documented and tested
- ✅ Can be enabled by default

---

**Status Legend:**
- [x] = Completed (pre-implemented)
- [ ] = To be verified on AWS
- ✅ = Verification passed
- ❌ = Issue found, needs fixing

---

**Next:** Run on AWS and check the boxes! 🚀
