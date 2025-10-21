#!/bin/bash
# Quick benchmark script for NVIDIA cuEquivariance Kernels
# Usage: ./benchmark_kernels.sh

set -e

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "   NVIDIA cuEquivariance Kernels Benchmark - Quick Run"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Configuration
NUM_SAMPLES=1  # Start with 1 sample for quick test
DATASET="abag_public"  # or asos_public
INPUT_JSONL="hackathon_data/datasets/${DATASET}/${DATASET}.jsonl"
MSA_DIR="hackathon_data/datasets/${DATASET}/msa/"

# Check if files exist
if [ ! -f "$INPUT_JSONL" ]; then
    echo "❌ Error: Input file not found: $INPUT_JSONL"
    echo "   Please run this script from the boltz-hackathon-template-dsc root directory"
    exit 1
fi

if [ ! -d "$MSA_DIR" ]; then
    echo "❌ Error: MSA directory not found: $MSA_DIR"
    exit 1
fi

echo "✅ Found dataset: $DATASET"
echo "✅ Input: $INPUT_JSONL"
echo "✅ MSA: $MSA_DIR"
echo ""

# Get first sample ID (datapoint_id from JSONL)
SAMPLE_ID=$(head -n 1 "$INPUT_JSONL" | python3 -c "import sys, json; print(json.load(sys.stdin)['datapoint_id'])")

echo "Testing with sample: $SAMPLE_ID"
echo ""

# Step 1: Baseline (WITHOUT kernels)
echo "────────────────────────────────────────────────────────────────────────────────"
echo "Step 1/3: Running BASELINE (WITHOUT NVIDIA Kernels)..."
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

OUTPUT_DIR_BASELINE="benchmark_baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR_BASELINE"

echo "Running Boltz prediction WITHOUT kernels..."
START_TIME_BASELINE=$(date +%s)

python -m boltz.main predict \
    "hackathon_data/datasets/${DATASET}/${SAMPLE_ID}" \
    --out_dir "$OUTPUT_DIR_BASELINE" \
    --recycling_steps 3 \
    --sampling_steps 50 \
    --step_scale 2.0 \
    --no_kernels

if [ $? -ne 0 ]; then
    echo "❌ Baseline benchmark failed!"
    exit 1
fi

END_TIME_BASELINE=$(date +%s)
DURATION_BASELINE=$((END_TIME_BASELINE - START_TIME_BASELINE))

echo ""
echo "✅ Baseline complete! Time: ${DURATION_BASELINE}s"
echo ""

# Step 2: Optimized (WITH kernels)
echo "────────────────────────────────────────────────────────────────────────────────"
echo "Step 2/3: Running OPTIMIZED (WITH NVIDIA Kernels)..."
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

OUTPUT_DIR_OPTIMIZED="benchmark_optimized_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR_OPTIMIZED"

echo "Running Boltz prediction WITH kernels..."
START_TIME_OPTIMIZED=$(date +%s)

python -m boltz.main predict \
    "hackathon_data/datasets/${DATASET}/${SAMPLE_ID}" \
    --out_dir "$OUTPUT_DIR_OPTIMIZED" \
    --recycling_steps 3 \
    --sampling_steps 50 \
    --step_scale 2.0

if [ $? -ne 0 ]; then
    echo "❌ Optimized benchmark failed!"
    exit 1
fi

END_TIME_OPTIMIZED=$(date +%s)
DURATION_OPTIMIZED=$((END_TIME_OPTIMIZED - START_TIME_OPTIMIZED))

echo ""
echo "✅ Optimized complete! Time: ${DURATION_OPTIMIZED}s"
echo ""

# Step 3: Compare and Report
echo "────────────────────────────────────────────────────────────────────────────────"
echo "Step 3/3: Generating comparison report..."
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

TIME_SAVED=$((DURATION_BASELINE - DURATION_OPTIMIZED))
SPEEDUP=$(python3 -c "print(round($DURATION_BASELINE / $DURATION_OPTIMIZED, 2))")
PERCENT_FASTER=$(python3 -c "print(round(($SPEEDUP - 1) * 100, 1))")

# Create results JSON
RESULTS_FILE="benchmark_results_$(date +%Y%m%d_%H%M%S).json"
cat > "$RESULTS_FILE" <<EOF
{
  "dataset": "$DATASET",
  "sample_id": "$SAMPLE_ID",
  "baseline_time_seconds": $DURATION_BASELINE,
  "optimized_time_seconds": $DURATION_OPTIMIZED,
  "time_saved_seconds": $TIME_SAVED,
  "speedup": $SPEEDUP,
  "percent_faster": $PERCENT_FASTER,
  "baseline_dir": "$OUTPUT_DIR_BASELINE",
  "optimized_dir": "$OUTPUT_DIR_OPTIMIZED",
  "timestamp": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "   BENCHMARK RESULTS"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Sample: $SAMPLE_ID"
echo ""
echo "Inference times:"
echo "  WITHOUT kernels (baseline): ${DURATION_BASELINE}s"
echo "  WITH kernels (optimized):   ${DURATION_OPTIMIZED}s"
echo ""
echo "Performance improvement:"
echo "  Time saved:     ${TIME_SAVED}s"
echo "  Speedup:        ${SPEEDUP}x"
echo "  Percent faster: ${PERCENT_FASTER}%"
echo ""

if (( $(echo "$SPEEDUP > 1.0" | bc -l) )); then
    echo "✅ SUCCESS: NVIDIA kernels provide ${PERCENT_FASTER}% speedup!"
else
    echo "⚠️  WARNING: No speedup observed."
    echo "   This might be due to:"
    echo "   - Short sequences (kernels optimize longer sequences)"
    echo "   - Reduced diffusion steps (50 vs 200)"
    echo "   - Cold cache / warmup needed"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Results saved to:"
echo "   - $RESULTS_FILE"
echo ""
echo "📁 Output directories:"
echo "   - $OUTPUT_DIR_BASELINE (without kernels)"
echo "   - $OUTPUT_DIR_OPTIMIZED (with kernels)"
echo ""
echo "Next steps:"
echo "   1. Review $RESULTS_FILE for detailed metrics"
echo "   2. Run with more samples: Edit NUM_SAMPLES variable in script"
echo "   3. Run with full diffusion: Remove --num_diffn_timesteps flag"
echo "   4. Compare protein structures in output directories"
echo ""
echo "To test with full dataset settings:"
echo "   ./benchmark_kernels.sh --full"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
