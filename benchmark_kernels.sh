#!/bin/bash
# Quick benchmark script for NVIDIA cuEquivariance Kernels
# Usage: ./benchmark_kernels.sh

set -e

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "   NVIDIA cuEquivariance Kernels Benchmark - Quick Run"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Configuration
DATASET="abag_public"

# Check if dataset exists
if [ ! -f "hackathon_data/datasets/${DATASET}/${DATASET}.jsonl" ]; then
    echo "❌ Error: Dataset not found"
    exit 1
fi

echo "✅ Found dataset: $DATASET"
echo ""

# Step 1: Baseline (WITHOUT kernels)
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "PHASE 1: Testing WITHOUT NVIDIA Kernels (Baseline)"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Running prediction with --no_kernels flag..."

cd hackathon

start_time=$(date +%s)

python predict_hackathon.py \
    --dataset "$DATASET" \
    --intermediate-dir ../benchmark_baseline \
    --submission-dir ../benchmark_baseline/submission \
    --num-samples 1 \
    -- --no_kernels --sampling_steps 50 --override

if [ $? -ne 0 ]; then
    echo "❌ Baseline prediction failed!"
    cd ..
    exit 1
fi

end_time=$(date +%s)
baseline_time=$((end_time - start_time))

cd ..

echo ""
echo "✅ Baseline complete: ${baseline_time}s"
echo ""

# Step 2: Optimized (WITH kernels)
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "PHASE 2: Testing WITH NVIDIA Kernels (Optimized)"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Running prediction WITH kernels (default)..."

cd hackathon

start_time=$(date +%s)

python predict_hackathon.py \
    --dataset "$DATASET" \
    --intermediate-dir ../benchmark_optimized \
    --submission-dir ../benchmark_optimized/submission \
    --num-samples 1 \
    -- --sampling_steps 50 --override

if [ $? -ne 0 ]; then
    echo "❌ Optimized prediction failed!"
    cd ..
    exit 1
fi

end_time=$(date +%s)
optimized_time=$((end_time - start_time))

cd ..

echo ""
echo "✅ Optimized complete: ${optimized_time}s"
echo ""

# Step 3: Compare
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "RESULTS SUMMARY"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

time_saved=$((baseline_time - optimized_time))
speedup=$(echo "scale=2; $baseline_time / $optimized_time" | bc)
percent_faster=$(echo "scale=1; ($speedup - 1) * 100" | bc)

echo "Dataset: $DATASET (1 sample)"
echo ""
echo "Total inference time:"
echo "  WITHOUT kernels (baseline): ${baseline_time}s"
echo "  WITH kernels (optimized):   ${optimized_time}s"
echo ""
echo "Performance improvement:"
echo "  Time saved:     ${time_saved}s"
echo "  Speedup:        ${speedup}x"
echo "  Percent faster: ${percent_faster}%"
echo ""

if (( $(echo "$speedup > 1.0" | bc -l) )); then
    echo "✅ SUCCESS: NVIDIA kernels provide ${percent_faster}% speedup!"
else
    echo "⚠️  WARNING: No significant speedup observed"
    echo "   Try with full settings: --sampling_steps 200"
fi

# Save results
cat > benchmark_results.json <<EOF
{
  "dataset": "$DATASET",
  "num_samples": 1,
  "baseline_time": $baseline_time,
  "optimized_time": $optimized_time,
  "speedup": $speedup,
  "time_saved": $time_saved,
  "percent_faster": $percent_faster,
  "configuration": {
    "recycling_steps": 3,
    "sampling_steps": 50
  },
  "directories": {
    "baseline": "benchmark_baseline",
    "optimized": "benchmark_optimized"
  }
}
EOF

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Results saved to: benchmark_results.json"
echo ""
echo "📁 Output directories:"
echo "   - benchmark_baseline/ (without kernels)"
echo "   - benchmark_optimized/ (with kernels)"
echo ""
echo "To test with full settings (200 sampling steps):"
echo "   Edit the script and change --sampling_steps 50 to 200"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"

