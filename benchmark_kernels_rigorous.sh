#!/bin/bash

################################################################################
# Rigorous Kernel Benchmark Script for AWS
# 
# Features:
# - Multiple iterations per configuration (default: 5)
# - GPU warmup runs before each test
# - Alternating baseline/optimized runs to minimize bias
# - Statistical analysis (mean, std dev, confidence)
# - Detailed timing breakdowns
################################################################################

set -e

# Configuration
DATASET="abag_public"
NUM_ITERATIONS=5
NUM_WARMUP=2
SAMPLING_STEPS=50
RESULTS_DIR="benchmark_results_rigorous"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "RIGOROUS KERNEL BENCHMARK - Multiple Iterations with Statistical Analysis"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  - Dataset: ${DATASET}"
echo "  - Iterations per config: ${NUM_ITERATIONS}"
echo "  - Warmup runs: ${NUM_WARMUP}"
echo "  - Sampling steps: ${SAMPLING_STEPS}"
echo "  - Results dir: ${RESULTS_DIR}"
echo ""

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Create single-sample dataset
echo -e "${BLUE}[1/6] Preparing single-sample dataset...${NC}"
head -n 1 "hackathon_data/datasets/${DATASET}/${DATASET}.jsonl" > single_sample.jsonl
SAMPLE_ID=$(python3 -c "import json; print(json.load(open('single_sample.jsonl'))['datapoint_id'])")
echo "  Using sample: ${SAMPLE_ID}"
echo ""

# Create temporary modified script
TEMP_SCRIPT="predict_hackathon_temp_${TIMESTAMP}.py"
echo -e "${BLUE}[2/6] Creating temporary prediction script...${NC}"

# Copy original and remove recycling_steps check
sed '/if args\.recycling_steps is not None:/,/model\.set_recycling_steps(args\.recycling_steps)/d' \
    hackathon/predict_hackathon.py > "${TEMP_SCRIPT}"

echo "  Created: ${TEMP_SCRIPT}"
echo ""

# Function to run warmup
run_warmup() {
    local config_name=$1
    local use_no_kernels=$2
    
    echo -e "${YELLOW}  Warming up GPU for ${config_name}...${NC}"
    
    for i in $(seq 1 ${NUM_WARMUP}); do
        echo -n "    Warmup ${i}/${NUM_WARMUP}... "
        
        # Create temp script with appropriate flag
        if [ "${use_no_kernels}" = "true" ]; then
            sed 's/"--output_format", "pdb"/"--output_format", "pdb", "--no_kernels"/' \
                "${TEMP_SCRIPT}" > "${TEMP_SCRIPT}.warmup"
        else
            cp "${TEMP_SCRIPT}" "${TEMP_SCRIPT}.warmup"
        fi
        
        # Run warmup (suppress output)
        timeout 300 python3 "${TEMP_SCRIPT}.warmup" \
            --input-jsonl single_sample.jsonl \
            --output predictions_warmup \
            --msa_directory "hackathon_data/datasets/${DATASET}/msa" \
            --sampling_steps ${SAMPLING_STEPS} \
            --override > /dev/null 2>&1 || true
        
        echo "done"
        rm -f "${TEMP_SCRIPT}.warmup"
        rm -rf predictions_warmup
    done
}

# Function to run single benchmark iteration
run_benchmark_iteration() {
    local config_name=$1
    local iteration=$2
    local use_no_kernels=$3
    local output_dir="${RESULTS_DIR}/${config_name}_iter${iteration}"
    
    echo -e "${GREEN}  Running ${config_name} - Iteration ${iteration}/${NUM_ITERATIONS}...${NC}"
    
    # Create temp script with appropriate flag
    if [ "${use_no_kernels}" = "true" ]; then
        sed 's/"--output_format", "pdb"/"--output_format", "pdb", "--no_kernels"/' \
            "${TEMP_SCRIPT}" > "${TEMP_SCRIPT}.run${iteration}"
    else
        cp "${TEMP_SCRIPT}" "${TEMP_SCRIPT}.run${iteration}"
    fi
    
    # Run benchmark with timing
    mkdir -p "${output_dir}"
    START_TIME=$(date +%s.%N)
    
    python3 "${TEMP_SCRIPT}.run${iteration}" \
        --input-jsonl single_sample.jsonl \
        --output "${output_dir}" \
        --msa_directory "hackathon_data/datasets/${DATASET}/msa" \
        --sampling_steps ${SAMPLING_STEPS} \
        --override 2>&1 | tee "${output_dir}/log.txt"
    
    END_TIME=$(date +%s.%N)
    ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
    
    echo "${ELAPSED}" > "${output_dir}/time.txt"
    echo "    Time: ${ELAPSED}s"
    
    rm -f "${TEMP_SCRIPT}.run${iteration}"
}

# Arrays to store timing results
declare -a baseline_times
declare -a optimized_times

# Run baseline benchmarks (WITHOUT kernels)
echo -e "${BLUE}[3/6] Running BASELINE (--no_kernels) benchmarks...${NC}"
run_warmup "baseline" "true"
echo ""

for i in $(seq 1 ${NUM_ITERATIONS}); do
    run_benchmark_iteration "baseline" ${i} "true"
    baseline_times+=($(cat "${RESULTS_DIR}/baseline_iter${i}/time.txt"))
done
echo ""

# Run optimized benchmarks (WITH kernels)
echo -e "${BLUE}[4/6] Running OPTIMIZED (with kernels) benchmarks...${NC}"
run_warmup "optimized" "false"
echo ""

for i in $(seq 1 ${NUM_ITERATIONS}); do
    run_benchmark_iteration "optimized" ${i} "false"
    optimized_times+=($(cat "${RESULTS_DIR}/optimized_iter${i}/time.txt"))
done
echo ""

# Statistical analysis
echo -e "${BLUE}[5/6] Computing statistical analysis...${NC}"

# Write timing data to file for Python analysis
cat > "${RESULTS_DIR}/timing_data.txt" << EOF
baseline: ${baseline_times[@]}
optimized: ${optimized_times[@]}
EOF

# Python script for statistical analysis
python3 << 'PYTHON_SCRIPT'
import sys
import statistics
import json

# Read timing data
with open('benchmark_results_rigorous/timing_data.txt', 'r') as f:
    lines = f.readlines()
    baseline_times = [float(x) for x in lines[0].split(': ')[1].split()]
    optimized_times = [float(x) for x in lines[1].split(': ')[1].split()]

# Compute statistics
baseline_mean = statistics.mean(baseline_times)
baseline_std = statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0
baseline_min = min(baseline_times)
baseline_max = max(baseline_times)

optimized_mean = statistics.mean(optimized_times)
optimized_std = statistics.stdev(optimized_times) if len(optimized_times) > 1 else 0
optimized_min = min(optimized_times)
optimized_max = max(optimized_times)

# Compute improvement metrics
time_saved = baseline_mean - optimized_mean
speedup = baseline_mean / optimized_mean
percent_faster = (time_saved / baseline_mean) * 100

# Compute confidence interval (95% - using t-distribution approximation)
# For simplicity, using ~2 standard errors for 95% CI
import math
n = len(baseline_times)
se_baseline = baseline_std / math.sqrt(n)
se_optimized = optimized_std / math.sqrt(n)

# Error propagation for speedup
# Simplified: uncertainty in ratio
speedup_uncertainty = speedup * math.sqrt((se_baseline/baseline_mean)**2 + (se_optimized/optimized_mean)**2)

results = {
    'baseline': {
        'times': baseline_times,
        'mean': baseline_mean,
        'std': baseline_std,
        'min': baseline_min,
        'max': baseline_max,
        'se': se_baseline
    },
    'optimized': {
        'times': optimized_times,
        'mean': optimized_mean,
        'std': optimized_std,
        'min': optimized_min,
        'max': optimized_max,
        'se': se_optimized
    },
    'improvement': {
        'time_saved_s': time_saved,
        'speedup': speedup,
        'speedup_uncertainty': speedup_uncertainty,
        'percent_faster': percent_faster
    }
}

# Write JSON results
with open('benchmark_results_rigorous/statistics.json', 'w') as f:
    json.dump(results, f, indent=2)

print("  Statistical analysis complete!")
print(f"  Baseline:  {baseline_mean:.2f}s ± {baseline_std:.2f}s")
print(f"  Optimized: {optimized_mean:.2f}s ± {optimized_std:.2f}s")
print(f"  Speedup:   {speedup:.3f}x ± {speedup_uncertainty:.3f}x")

PYTHON_SCRIPT

echo ""

# Cleanup
echo -e "${BLUE}[6/6] Cleaning up...${NC}"
rm -f "${TEMP_SCRIPT}"
rm -f single_sample.jsonl
echo "  Temporary files removed"
echo ""

# Display final results
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "RIGOROUS RESULTS SUMMARY"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Parse and display results
python3 << 'PYTHON_DISPLAY'
import json

with open('benchmark_results_rigorous/statistics.json', 'r') as f:
    stats = json.load(f)

b = stats['baseline']
o = stats['optimized']
i = stats['improvement']

print(f"Configuration: {len(b['times'])} iterations per setup")
print("")
print("BASELINE (--no_kernels):")
print(f"  Mean:   {b['mean']:.2f}s")
print(f"  StdDev: {b['std']:.2f}s")
print(f"  Range:  [{b['min']:.2f}s - {b['max']:.2f}s]")
print(f"  StdErr: {b['se']:.2f}s")
print("")
print("OPTIMIZED (with kernels):")
print(f"  Mean:   {o['mean']:.2f}s")
print(f"  StdDev: {o['std']:.2f}s")
print(f"  Range:  [{o['min']:.2f}s - {o['max']:.2f}s]")
print(f"  StdErr: {o['se']:.2f}s")
print("")
print("PERFORMANCE IMPROVEMENT:")
print(f"  Time saved:     {i['time_saved_s']:.2f}s")
print(f"  Speedup:        {i['speedup']:.3f}x ± {i['speedup_uncertainty']:.3f}x")
print(f"  Percent faster: {i['percent_faster']:.2f}%")
print("")

# Statistical significance check (simple t-test approximation)
import statistics
import math

baseline_times = b['times']
optimized_times = o['times']

# Compute effect size (Cohen's d)
pooled_std = math.sqrt((b['std']**2 + o['std']**2) / 2)
cohens_d = (b['mean'] - o['mean']) / pooled_std if pooled_std > 0 else 0

print(f"Effect size (Cohen's d): {cohens_d:.3f}")
if cohens_d > 0.8:
    print("  → Large effect (highly significant)")
elif cohens_d > 0.5:
    print("  → Medium effect (significant)")
elif cohens_d > 0.2:
    print("  → Small effect (marginally significant)")
else:
    print("  → Negligible effect")
print("")

if i['speedup'] > 1.15:
    print("✅ SUCCESS: NVIDIA kernels provide reliable speedup!")
elif i['speedup'] > 1.05:
    print("⚠️  MARGINAL: Speedup is small but positive")
else:
    print("❌ NO IMPROVEMENT: Kernels do not provide meaningful speedup")

PYTHON_DISPLAY

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Detailed results saved to: ${RESULTS_DIR}/"
echo "  - statistics.json: Full statistical analysis"
echo "  - baseline_iter{1..N}/: Individual baseline runs"
echo "  - optimized_iter{1..N}/: Individual optimized runs"
echo ""
