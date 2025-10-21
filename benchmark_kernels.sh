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
JSONL_FILE="hackathon_data/datasets/${DATASET}/${DATASET}.jsonl"

# Check if dataset exists
if [ ! -f "$JSONL_FILE" ]; then
    echo "❌ Error: Dataset not found"
    exit 1
fi

echo "✅ Found dataset: $DATASET"
echo ""

# Create single sample JSONL (first line only)
mkdir -p benchmark_temp
head -n 1 "$JSONL_FILE" > benchmark_temp/single_sample.jsonl

echo "Testing with 1 sample (50 sampling steps for speed)"
echo ""

# Step 1: Baseline (WITHOUT kernels)
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "PHASE 1: Testing WITHOUT NVIDIA Kernels (Baseline)"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Temporarily modify predict_hackathon.py to add --no_kernels
cd hackathon
cp predict_hackathon.py predict_hackathon_baseline.py
# Replace the boltz command line to add flags
sed -i 's/"--output_format", "pdb",/"--output_format", "pdb", "--no_kernels", "--sampling_steps", "50", "--override",/' predict_hackathon_baseline.py
# Remove the recycling_steps check since we're not using it
sed -i '/if args.recycling_steps/,/fixed.extend/d' predict_hackathon_baseline.py

start_time=$(date +%s)

python predict_hackathon_baseline.py \
    --input-jsonl "../benchmark_temp/single_sample.jsonl" \
    --msa-dir "../hackathon_data/datasets/${DATASET}/msa" \
    --intermediate-dir ../benchmark_baseline \
    --submission-dir ../benchmark_baseline/submission

if [ $? -ne 0 ]; then
    echo "❌ Baseline prediction failed!"
    rm predict_hackathon_baseline.py
    cd ..
    exit 1
fi

end_time=$(date +%s)
baseline_time=$((end_time - start_time))

rm predict_hackathon_baseline.py
cd ..

echo ""
echo "✅ Baseline complete: ${baseline_time}s"
echo ""

# Step 2: Optimized (WITH kernels)
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "PHASE 2: Testing WITH NVIDIA Kernels (Optimized)"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Temporarily modify predict_hackathon.py to add --sampling_steps
cd hackathon
cp predict_hackathon.py predict_hackathon_optimized.py
# Replace the boltz command line to add flags (no --no_kernels for optimized!)
sed -i 's/"--output_format", "pdb",/"--output_format", "pdb", "--sampling_steps", "50", "--override",/' predict_hackathon_optimized.py
# Remove the recycling_steps check since we're not using it
sed -i '/if args.recycling_steps/,/fixed.extend/d' predict_hackathon_optimized.py

start_time=$(date +%s)

python predict_hackathon_optimized.py \
    --input-jsonl "../benchmark_temp/single_sample.jsonl" \
    --msa-dir "../hackathon_data/datasets/${DATASET}/msa" \
    --intermediate-dir ../benchmark_optimized \
    --submission-dir ../benchmark_optimized/submission

if [ $? -ne 0 ]; then
    echo "❌ Optimized prediction failed!"
    rm predict_hackathon_optimized.py
    cd ..
    exit 1
fi

end_time=$(date +%s)
optimized_time=$((end_time - start_time))

rm predict_hackathon_optimized.py
cd ..

echo ""
echo "✅ Optimized complete: ${optimized_time}s"
echo ""

# Cleanup temp files
rm -rf benchmark_temp

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
    echo "   Try with full settings: Change sampling_steps to 200 in script"
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
echo "   Edit the script and change sampling_steps from 50 to 200"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"


