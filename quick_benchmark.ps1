# Quick benchmark script for Adaptive Recycling (PowerShell version)
# Usage: .\quick_benchmark.ps1

$ErrorActionPreference = "Stop"

Write-Host "════════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   Adaptive Recycling Benchmark - Quick Run (25% of dataset)" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Configuration
$DATASET = "abag_public"  # or asos_public
$INPUT_JSONL = "hackathon_data/datasets/$DATASET/$DATASET.jsonl"
$MSA_DIR = "hackathon_data/datasets/$DATASET/msa/"
$RECYCLING_STEPS = 6  # Testing with 6 recycling steps (vs Boltz default of 3)

# Calculate 25% of the dataset (with minimum of 2 samples)
$TOTAL_SAMPLES = (Get-Content $INPUT_JSONL | Measure-Object -Line).Lines
$NUM_SAMPLES = [Math]::Max(2, [Math]::Round($TOTAL_SAMPLES * 0.25))
Write-Host "📊 Dataset has $TOTAL_SAMPLES samples, using $NUM_SAMPLES (25%)" -ForegroundColor Green
Write-Host "🔄 Using $RECYCLING_STEPS recycling steps (Boltz default: 3)" -ForegroundColor Green
Write-Host ""

# Check if files exist
if (-not (Test-Path $INPUT_JSONL)) {
    Write-Host "❌ Error: Input file not found: $INPUT_JSONL" -ForegroundColor Red
    Write-Host "   Please run this script from the boltz-hackathon-template-dsc root directory" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $MSA_DIR)) {
    Write-Host "❌ Error: MSA directory not found: $MSA_DIR" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Found dataset: $DATASET" -ForegroundColor Green
Write-Host "✅ Input: $INPUT_JSONL" -ForegroundColor Green
Write-Host "✅ MSA: $MSA_DIR" -ForegroundColor Green
Write-Host ""

# Step 1: Baseline
Write-Host "────────────────────────────────────────────────────────────────────────────────" -ForegroundColor Yellow
Write-Host "Step 1/3: Running BASELINE (without optimization)..." -ForegroundColor Yellow
Write-Host "────────────────────────────────────────────────────────────────────────────────" -ForegroundColor Yellow
Write-Host ""

python benchmark_adaptive_recycling.py `
    --mode baseline `
    --input-jsonl "$INPUT_JSONL" `
    --msa-dir "$MSA_DIR" `
    --output-dir ./benchmark_baseline `
    --num-samples $NUM_SAMPLES `
    --recycling-steps $RECYCLING_STEPS

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Baseline benchmark failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ Baseline complete!" -ForegroundColor Green
Write-Host ""

# Step 2: Optimized
Write-Host "────────────────────────────────────────────────────────────────────────────────" -ForegroundColor Yellow
Write-Host "Step 2/3: Running OPTIMIZED (with Adaptive Recycling)..." -ForegroundColor Yellow
Write-Host "────────────────────────────────────────────────────────────────────────────────" -ForegroundColor Yellow
Write-Host ""

python benchmark_adaptive_recycling.py `
    --mode optimized `
    --input-jsonl "$INPUT_JSONL" `
    --msa-dir "$MSA_DIR" `
    --output-dir ./benchmark_optimized `
    --num-samples $NUM_SAMPLES `
    --recycling-steps $RECYCLING_STEPS

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Optimized benchmark failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ Optimized complete!" -ForegroundColor Green
Write-Host ""

# Step 3: Compare
Write-Host "────────────────────────────────────────────────────────────────────────────────" -ForegroundColor Yellow
Write-Host "Step 3/3: Generating comparison report..." -ForegroundColor Yellow
Write-Host "────────────────────────────────────────────────────────────────────────────────" -ForegroundColor Yellow
Write-Host ""

python benchmark_adaptive_recycling.py `
    --mode compare `
    --baseline-dir ./benchmark_baseline `
    --optimized-dir ./benchmark_optimized

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Comparison failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   ✅ BENCHMARK COMPLETE!" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Results saved to:" -ForegroundColor Cyan
Write-Host "   - benchmark_comparison_report.json (detailed)" -ForegroundColor White
Write-Host "   - benchmark_comparison_report.csv (summary)" -ForegroundColor White
Write-Host ""
Write-Host "📁 Output directories:" -ForegroundColor Cyan
Write-Host "   - benchmark_baseline/ (without optimization)" -ForegroundColor White
Write-Host "   - benchmark_optimized/ (with optimization)" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "   1. Review benchmark_comparison_report.json" -ForegroundColor White
Write-Host "   2. Check logs in benchmark_*/prediction.log" -ForegroundColor White
Write-Host "   3. Compare structures in benchmark_*/submission/" -ForegroundColor White
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
