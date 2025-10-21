#!/bin/bash
# Full benchmark script for NVIDIA cuEquivariance Kernels
# Usage: ./benchmark_kernels_full.sh [OPTIONS]

set -e

# Default configuration
NUM_SAMPLES=1
DATASET="abag_public"
NUM_DIFFUSION_STEPS=50  # Use 200 for full production settings
RECYCLING_STEPS=3
NUM_RUNS=1  # Run each test multiple times for averaging

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --full)
            NUM_DIFFUSION_STEPS=200
            echo "Using full production settings (200 diffusion steps)"
            shift
            ;;
        --runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./benchmark_kernels_full.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --samples N      Number of samples to test (default: 1)"
            echo "  --dataset NAME   Dataset to use: abag_public or asos_public (default: abag_public)"
            echo "  --full           Use full production settings (200 diffusion steps)"
            echo "  --runs N         Number of runs per configuration for averaging (default: 1)"
            echo "  --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./benchmark_kernels_full.sh --samples 3"
            echo "  ./benchmark_kernels_full.sh --full --runs 2"
            echo "  ./benchmark_kernels_full.sh --dataset asos_public --samples 2"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "   NVIDIA cuEquivariance Kernels Benchmark"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Dataset:          $DATASET"
echo "  Samples:          $NUM_SAMPLES"
echo "  Diffusion steps:  $NUM_DIFFUSION_STEPS"
echo "  Recycling steps:  $RECYCLING_STEPS"
echo "  Runs per config:  $NUM_RUNS"
echo ""

INPUT_JSONL="hackathon_data/datasets/${DATASET}/${DATASET}.jsonl"
MSA_DIR="hackathon_data/datasets/${DATASET}/msa/"

# Check if files exist
if [ ! -f "$INPUT_JSONL" ]; then
    echo "❌ Error: Input file not found: $INPUT_JSONL"
    exit 1
fi

if [ ! -d "$MSA_DIR" ]; then
    echo "❌ Error: MSA directory not found: $MSA_DIR"
    exit 1
fi

# Get sample IDs
echo "Reading sample IDs from $INPUT_JSONL..."
SAMPLE_IDS=()
count=0
while IFS= read -r line && [ $count -lt $NUM_SAMPLES ]; do
    sample_id=$(echo "$line" | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")
    SAMPLE_IDS+=("$sample_id")
    ((count++))
done < "$INPUT_JSONL"

echo "Testing with samples: ${SAMPLE_IDS[*]}"
echo ""

# Arrays to store timing results
BASELINE_TIMES=()
OPTIMIZED_TIMES=()

# Step 1: Baseline (WITHOUT kernels)
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "PHASE 1: Testing WITHOUT NVIDIA Kernels (Baseline)"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

OUTPUT_DIR_BASELINE="benchmark_baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR_BASELINE"

for sample_id in "${SAMPLE_IDS[@]}"; do
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo "Sample: $sample_id (WITHOUT kernels)"
    echo "────────────────────────────────────────────────────────────────────────────────"
    
    total_time=0
    
    for ((run=1; run<=NUM_RUNS; run++)); do
        echo "Run $run/$NUM_RUNS..."
        
        start_time=$(date +%s.%N)
        
        python -m boltz.main predict \
            "hackathon_data/datasets/${DATASET}/${sample_id}" \
            --out_dir "$OUTPUT_DIR_BASELINE/${sample_id}" \
            --recycling_steps $RECYCLING_STEPS \
            --num_diffn_timesteps $NUM_DIFFUSION_STEPS \
            --step_scale 2.0 \
            --override \
            --no_kernels > /dev/null 2>&1
        
        if [ $? -ne 0 ]; then
            echo "❌ Baseline prediction failed for $sample_id!"
            exit 1
        fi
        
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc)
        total_time=$(echo "$total_time + $duration" | bc)
        
        echo "  Time: ${duration}s"
    done
    
    avg_time=$(echo "scale=2; $total_time / $NUM_RUNS" | bc)
    BASELINE_TIMES+=("$avg_time")
    
    echo "✅ Average time: ${avg_time}s"
    echo ""
done

echo "✅ Baseline phase complete!"
echo ""

# Step 2: Optimized (WITH kernels)
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "PHASE 2: Testing WITH NVIDIA Kernels (Optimized)"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

OUTPUT_DIR_OPTIMIZED="benchmark_optimized_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR_OPTIMIZED"

for sample_id in "${SAMPLE_IDS[@]}"; do
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo "Sample: $sample_id (WITH kernels)"
    echo "────────────────────────────────────────────────────────────────────────────────"
    
    total_time=0
    
    for ((run=1; run<=NUM_RUNS; run++)); do
        echo "Run $run/$NUM_RUNS..."
        
        start_time=$(date +%s.%N)
        
        python -m boltz.main predict \
            "hackathon_data/datasets/${DATASET}/${sample_id}" \
            --out_dir "$OUTPUT_DIR_OPTIMIZED/${sample_id}" \
            --recycling_steps $RECYCLING_STEPS \
            --num_diffn_timesteps $NUM_DIFFUSION_STEPS \
            --step_scale 2.0 \
            --override > /dev/null 2>&1
        
        if [ $? -ne 0 ]; then
            echo "❌ Optimized prediction failed for $sample_id!"
            exit 1
        fi
        
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc)
        total_time=$(echo "$total_time + $duration" | bc)
        
        echo "  Time: ${duration}s"
    done
    
    avg_time=$(echo "scale=2; $total_time / $NUM_RUNS" | bc)
    OPTIMIZED_TIMES+=("$avg_time")
    
    echo "✅ Average time: ${avg_time}s"
    echo ""
done

echo "✅ Optimized phase complete!"
echo ""

# Step 3: Compare and Report
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "RESULTS SUMMARY"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Calculate averages
total_baseline=0
total_optimized=0

for i in "${!BASELINE_TIMES[@]}"; do
    total_baseline=$(echo "$total_baseline + ${BASELINE_TIMES[$i]}" | bc)
    total_optimized=$(echo "$total_optimized + ${OPTIMIZED_TIMES[$i]}" | bc)
done

avg_baseline=$(echo "scale=2; $total_baseline / ${#BASELINE_TIMES[@]}" | bc)
avg_optimized=$(echo "scale=2; $total_optimized / ${#OPTIMIZED_TIMES[@]}" | bc)
speedup=$(echo "scale=2; $avg_baseline / $avg_optimized" | bc)
time_saved=$(echo "scale=2; $avg_baseline - $avg_optimized" | bc)
percent_faster=$(echo "scale=1; ($speedup - 1) * 100" | bc)

echo "Average inference time per sample:"
echo "  WITHOUT kernels (baseline): ${avg_baseline}s"
echo "  WITH kernels (optimized):   ${avg_optimized}s"
echo ""
echo "Performance improvement:"
echo "  Time saved:     ${time_saved}s per sample"
echo "  Speedup:        ${speedup}x"
echo "  Percent faster: ${percent_faster}%"
echo ""

if (( $(echo "$speedup > 1.0" | bc -l) )); then
    echo "✅ SUCCESS: NVIDIA kernels provide ${percent_faster}% speedup!"
else
    echo "⚠️  WARNING: No significant speedup observed"
fi

echo ""
echo "Per-sample breakdown:"
printf "%-20s %-18s %-18s %-10s\n" "Sample" "Baseline (s)" "Optimized (s)" "Speedup"
echo "────────────────────────────────────────────────────────────────────────────────"

for i in "${!SAMPLE_IDS[@]}"; do
    sample_speedup=$(echo "scale=2; ${BASELINE_TIMES[$i]} / ${OPTIMIZED_TIMES[$i]}" | bc)
    printf "%-20s %-18s %-18s %-10s\n" \
        "${SAMPLE_IDS[$i]}" \
        "${BASELINE_TIMES[$i]}" \
        "${OPTIMIZED_TIMES[$i]}" \
        "${sample_speedup}x"
done

# Save results to JSON
RESULTS_FILE="benchmark_results_$(date +%Y%m%d_%H%M%S).json"

cat > "$RESULTS_FILE" <<EOF
{
  "configuration": {
    "dataset": "$DATASET",
    "num_samples": $NUM_SAMPLES,
    "diffusion_steps": $NUM_DIFFUSION_STEPS,
    "recycling_steps": $RECYCLING_STEPS,
    "num_runs": $NUM_RUNS
  },
  "samples": [
EOF

for i in "${!SAMPLE_IDS[@]}"; do
    sample_speedup=$(echo "scale=2; ${BASELINE_TIMES[$i]} / ${OPTIMIZED_TIMES[$i]}" | bc)
    
    cat >> "$RESULTS_FILE" <<EOF
    {
      "id": "${SAMPLE_IDS[$i]}",
      "baseline_time": ${BASELINE_TIMES[$i]},
      "optimized_time": ${OPTIMIZED_TIMES[$i]},
      "speedup": $sample_speedup
    }$([ $i -lt $((${#SAMPLE_IDS[@]} - 1)) ] && echo "," || echo "")
EOF
done

cat >> "$RESULTS_FILE" <<EOF
  ],
  "summary": {
    "avg_baseline_time": $avg_baseline,
    "avg_optimized_time": $avg_optimized,
    "avg_speedup": $speedup,
    "time_saved": $time_saved,
    "percent_faster": $percent_faster
  },
  "directories": {
    "baseline": "$OUTPUT_DIR_BASELINE",
    "optimized": "$OUTPUT_DIR_OPTIMIZED"
  },
  "timestamp": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Results saved to: $RESULTS_FILE"
echo ""
echo "📁 Output directories:"
echo "   - $OUTPUT_DIR_BASELINE (without kernels)"
echo "   - $OUTPUT_DIR_OPTIMIZED (with kernels)"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
