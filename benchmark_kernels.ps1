# Quick benchmark script for NVIDIA cuEquivariance Kernels
# Usage: .\benchmark_kernels.ps1

$ErrorActionPreference = "Stop"

Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   NVIDIA cuEquivariance Kernels Benchmark - Quick Run" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Configuration
$NUM_SAMPLES = 1  # Start with 1 sample for quick test
$DATASET = "abag_public"  # or asos_public
$INPUT_JSONL = "hackathon_data/datasets/$DATASET/$DATASET.jsonl"
$MSA_DIR = "hackathon_data/datasets/$DATASET/msa/"

# Check if files exist
if (-not (Test-Path $INPUT_JSONL)) {
    Write-Host "❌ Error: Input file not found: $INPUT_JSONL" -ForegroundColor Red
    Write-Host "   Please run this script from the boltz-hackathon-template-dsc root directory" -ForegroundColor Red
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

# Get first sample ID
$firstLine = Get-Content $INPUT_JSONL -First 1
$sampleData = $firstLine | ConvertFrom-Json
$SAMPLE_ID = $sampleData.id

Write-Host "Testing with sample: $SAMPLE_ID" -ForegroundColor Yellow
Write-Host ""

# Step 1: Baseline (WITHOUT kernels)
Write-Host "────────────────────────────────────────────────────────────────────────────────" -ForegroundColor Cyan
Write-Host "Step 1/3: Running BASELINE (WITHOUT NVIDIA Kernels)..." -ForegroundColor Cyan
Write-Host "────────────────────────────────────────────────────────────────────────────────" -ForegroundColor Cyan
Write-Host ""

$outputDirBaseline = "benchmark_baseline_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Force -Path $outputDirBaseline | Out-Null

Write-Host "Running Boltz prediction WITHOUT kernels..." -ForegroundColor Yellow
$startTimeBaseline = Get-Date

try {
    python -m boltz.main predict `
        "hackathon_data/datasets/$DATASET/$SAMPLE_ID" `
        --out_dir $outputDirBaseline `
        --recycling_steps 3 `
        --num_diffn_timesteps 50 `
        --step_scale 2.0 `
        --no_kernels
    
    if ($LASTEXITCODE -ne 0) {
        throw "Baseline prediction failed"
    }
} catch {
    Write-Host "❌ Baseline benchmark failed: $_" -ForegroundColor Red
    exit 1
}

$endTimeBaseline = Get-Date
$durationBaseline = ($endTimeBaseline - $startTimeBaseline).TotalSeconds

Write-Host ""
Write-Host "✅ Baseline complete! Time: $([math]::Round($durationBaseline, 2))s" -ForegroundColor Green
Write-Host ""

# Step 2: Optimized (WITH kernels)
Write-Host "────────────────────────────────────────────────────────────────────────────────" -ForegroundColor Cyan
Write-Host "Step 2/3: Running OPTIMIZED (WITH NVIDIA Kernels)..." -ForegroundColor Cyan
Write-Host "────────────────────────────────────────────────────────────────────────────────" -ForegroundColor Cyan
Write-Host ""

$outputDirOptimized = "benchmark_optimized_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Force -Path $outputDirOptimized | Out-Null

Write-Host "Running Boltz prediction WITH kernels..." -ForegroundColor Yellow
$startTimeOptimized = Get-Date

try {
    python -m boltz.main predict `
        "hackathon_data/datasets/$DATASET/$SAMPLE_ID" `
        --out_dir $outputDirOptimized `
        --recycling_steps 3 `
        --num_diffn_timesteps 50 `
        --step_scale 2.0
    
    if ($LASTEXITCODE -ne 0) {
        throw "Optimized prediction failed"
    }
} catch {
    Write-Host "❌ Optimized benchmark failed: $_" -ForegroundColor Red
    exit 1
}

$endTimeOptimized = Get-Date
$durationOptimized = ($endTimeOptimized - $startTimeOptimized).TotalSeconds

Write-Host ""
Write-Host "✅ Optimized complete! Time: $([math]::Round($durationOptimized, 2))s" -ForegroundColor Green
Write-Host ""

# Step 3: Compare and Report
Write-Host "────────────────────────────────────────────────────────────────────────────────" -ForegroundColor Cyan
Write-Host "Step 3/3: Generating comparison report..." -ForegroundColor Cyan
Write-Host "────────────────────────────────────────────────────────────────────────────────" -ForegroundColor Cyan
Write-Host ""

$speedup = $durationBaseline / $durationOptimized
$timeSaved = $durationBaseline - $durationOptimized
$percentFaster = ($speedup - 1) * 100

# Create results object
$results = @{
    dataset = $DATASET
    sample_id = $SAMPLE_ID
    baseline_time_seconds = [math]::Round($durationBaseline, 2)
    optimized_time_seconds = [math]::Round($durationOptimized, 2)
    time_saved_seconds = [math]::Round($timeSaved, 2)
    speedup = [math]::Round($speedup, 2)
    percent_faster = [math]::Round($percentFaster, 1)
    baseline_dir = $outputDirBaseline
    optimized_dir = $outputDirOptimized
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
}

# Save to JSON
$resultsFile = "benchmark_results_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
$results | ConvertTo-Json -Depth 10 | Out-File -FilePath $resultsFile -Encoding utf8

Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   BENCHMARK RESULTS" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Sample: $SAMPLE_ID" -ForegroundColor Yellow
Write-Host ""
Write-Host "Inference times:" -ForegroundColor White
Write-Host "  WITHOUT kernels (baseline): $([math]::Round($durationBaseline, 2))s" -ForegroundColor White
Write-Host "  WITH kernels (optimized):   $([math]::Round($durationOptimized, 2))s" -ForegroundColor White
Write-Host ""
Write-Host "Performance improvement:" -ForegroundColor White
Write-Host "  Time saved:     $([math]::Round($timeSaved, 2))s" -ForegroundColor Green
Write-Host "  Speedup:        $([math]::Round($speedup, 2))x" -ForegroundColor Green
Write-Host "  Percent faster: $([math]::Round($percentFaster, 1))%" -ForegroundColor Green
Write-Host ""

if ($speedup -gt 1.0) {
    Write-Host "✅ SUCCESS: NVIDIA kernels provide $([math]::Round($percentFaster, 1))% speedup!" -ForegroundColor Green
} else {
    Write-Host "⚠️  WARNING: No speedup observed." -ForegroundColor Yellow
    Write-Host "   This might be due to:" -ForegroundColor Yellow
    Write-Host "   - Short sequences (kernels optimize longer sequences)" -ForegroundColor Yellow
    Write-Host "   - Reduced diffusion steps (50 vs 200)" -ForegroundColor Yellow
    Write-Host "   - Cold cache / warmup needed" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Results saved to:" -ForegroundColor White
Write-Host "   - $resultsFile" -ForegroundColor Cyan
Write-Host ""
Write-Host "📁 Output directories:" -ForegroundColor White
Write-Host "   - $outputDirBaseline (without kernels)" -ForegroundColor Cyan
Write-Host "   - $outputDirOptimized (with kernels)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "   1. Review $resultsFile for detailed metrics" -ForegroundColor Cyan
Write-Host "   2. Run with more samples: Edit NUM_SAMPLES variable in script" -ForegroundColor Cyan
Write-Host "   3. Run with full diffusion: Remove --num_diffn_timesteps flag" -ForegroundColor Cyan
Write-Host "   4. Compare protein structures in output directories" -ForegroundColor Cyan
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
