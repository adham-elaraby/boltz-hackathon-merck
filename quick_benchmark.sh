#!/bin/bash
# Quick benchmark script for Adaptive Recycling
# Usage: ./quick_benchmark.sh

set -e

echo "════════════════════════════════════════════════════════════════════════════════"
echo "   Adaptive Recycling Benchmark - Quick Run (25% of dataset)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Configuration
NUM_SAMPLES=5
DATASET="abag_public"  # or asos_public
INPUT_JSONL="hackathon_data/datasets/${DATASET}/${DATASET}.jsonl"
MSA_DIR="hackathon_data/datasets/${DATASET}/msa/"
RECYCLING_STEPS=6  # Testing with 6 recycling steps (vs Boltz default of 3)

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

# Step 1: Baseline
echo "────────────────────────────────────────────────────────────────────────────────"
echo "Step 1/3: Running BASELINE (without optimization)..."
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

python benchmark_adaptive_recycling.py \
    --mode baseline \
    --input-jsonl "$INPUT_JSONL" \
    --msa-dir "$MSA_DIR" \
    --output-dir ./benchmark_baseline \
    --num-samples $NUM_SAMPLES \
    --recycling-steps $RECYCLING_STEPS

if [ $? -ne 0 ]; then
    echo "❌ Baseline benchmark failed!"
    exit 1
fi

echo ""
echo "✅ Baseline complete!"
echo ""

# Step 2: Optimized
echo "────────────────────────────────────────────────────────────────────────────────"
echo "Step 2/3: Running OPTIMIZED (with Adaptive Recycling)..."
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

python benchmark_adaptive_recycling.py \
    --mode optimized \
    --input-jsonl "$INPUT_JSONL" \
    --msa-dir "$MSA_DIR" \
    --output-dir ./benchmark_optimized \
    --num-samples $NUM_SAMPLES \
    --recycling-steps $RECYCLING_STEPS

if [ $? -ne 0 ]; then
    echo "❌ Optimized benchmark failed!"
    exit 1
fi

echo ""
echo "✅ Optimized complete!"
echo ""

# Step 3: Compare
echo "────────────────────────────────────────────────────────────────────────────────"
echo "Step 3/3: Generating comparison report..."
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

python benchmark_adaptive_recycling.py \
    --mode compare \
    --baseline-dir ./benchmark_baseline \
    --optimized-dir ./benchmark_optimized

if [ $? -ne 0 ]; then
    echo "❌ Comparison failed!"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "   ✅ BENCHMARK COMPLETE!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Results saved to:"
echo "   - benchmark_comparison_report.json (detailed)"
echo "   - benchmark_comparison_report.csv (summary)"
echo ""
echo "📁 Output directories:"
echo "   - benchmark_baseline/ (without optimization)"
echo "   - benchmark_optimized/ (with optimization)"
echo ""
echo "Next steps:"
echo "   1. Review benchmark_comparison_report.json"
echo "   2. Check logs in benchmark_*/prediction.log"
echo "   3. Compare structures in benchmark_*/submission/"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
