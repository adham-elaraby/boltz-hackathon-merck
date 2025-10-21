#!/bin/bash

################################################################################
# Kernel Breakdown Benchmark - Isolate Contribution of Each Optimization
# 
# Tests 3 configurations:
# 1. BASELINE: --no_kernels (no optimizations)
# 2. TRIANGLE ONLY: Triangle kernels enabled, AttentionPairBias disabled
# 3. FULL OPTIMIZED: All kernels enabled (triangle + attention)
################################################################################

set -e

# Configuration
DATASET="abag_public"
NUM_ITERATIONS=3  # Fewer iterations since we're running 3 configs
NUM_WARMUP=1
SAMPLING_STEPS=50
RESULTS_DIR="benchmark_breakdown"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "KERNEL BREAKDOWN BENCHMARK - Isolate Each Optimization's Contribution"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  - Dataset: ${DATASET}"
echo "  - Iterations per config: ${NUM_ITERATIONS}"
echo "  - Warmup runs: ${NUM_WARMUP}"
echo "  - Sampling steps: ${SAMPLING_STEPS}"
echo "  - Results dir: ${RESULTS_DIR}"
echo ""
echo "Testing 3 configurations:"
echo "  1. BASELINE:       --no_kernels (PyTorch only)"
echo "  2. TRIANGLE ONLY:  Triangle kernels ON, Attention kernel OFF"
echo "  3. FULL OPTIMIZED: All kernels ON (triangle + attention)"
echo ""

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Prepare dataset
echo -e "${BLUE}[1/4] Preparing dataset...${NC}"
mkdir -p benchmark_temp_breakdown
head -n 1 "hackathon_data/datasets/${DATASET}/${DATASET}.jsonl" > benchmark_temp_breakdown/single_sample.jsonl
SAMPLE_ID=$(python3 -c "import json; print(json.load(open('benchmark_temp_breakdown/single_sample.jsonl'))['datapoint_id'])")
echo "  Using sample: ${SAMPLE_ID}"
echo ""

# Function to run warmup
run_warmup() {
    local config_name=$1
    local kernel_flags=$2
    
    echo -e "${YELLOW}  Warming up GPU for ${config_name}...${NC}"
    
    cd hackathon
    cp predict_hackathon.py predict_hackathon_warmup.py
    
    # Inject flags
    sed -i 's/"--output_format", "pdb",/"--output_format", "pdb", '"${kernel_flags}"', "--sampling_steps", "'${SAMPLING_STEPS}'", "--override",/' predict_hackathon_warmup.py
    sed -i '/if args.recycling_steps/,/fixed.extend/d' predict_hackathon_warmup.py
    
    timeout 300 python predict_hackathon_warmup.py \
        --input-jsonl "../benchmark_temp_breakdown/single_sample.jsonl" \
        --msa-dir "../hackathon_data/datasets/${DATASET}/msa" \
        --intermediate-dir ../warmup_temp \
        --submission-dir ../warmup_temp/submission > /dev/null 2>&1 || true
    
    rm predict_hackathon_warmup.py
    cd ..
    rm -rf warmup_temp
    
    echo "    Warmup complete"
}

# Function to run benchmark iteration
run_benchmark_iteration() {
    local config_name=$1
    local iteration=$2
    local kernel_flags=$3
    local output_dir="${RESULTS_DIR}/${config_name}_iter${iteration}"
    
    echo -e "${GREEN}  Running ${config_name} - Iteration ${iteration}/${NUM_ITERATIONS}...${NC}"
    
    cd hackathon
    cp predict_hackathon.py predict_hackathon_iter${iteration}.py
    
    # Inject flags
    sed -i 's/"--output_format", "pdb",/"--output_format", "pdb", '"${kernel_flags}"', "--sampling_steps", "'${SAMPLING_STEPS}'", "--override",/' predict_hackathon_iter${iteration}.py
    sed -i '/if args.recycling_steps/,/fixed.extend/d' predict_hackathon_iter${iteration}.py
    
    mkdir -p "../${output_dir}"
    START_TIME=$(date +%s.%N)
    
    python predict_hackathon_iter${iteration}.py \
        --input-jsonl "../benchmark_temp_breakdown/single_sample.jsonl" \
        --msa-dir "../hackathon_data/datasets/${DATASET}/msa" \
        --intermediate-dir "../${output_dir}" \
        --submission-dir "../${output_dir}/submission" > "../${output_dir}/log.txt" 2>&1
    
    END_TIME=$(date +%s.%N)
    ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
    
    echo "${ELAPSED}" > "../${output_dir}/time.txt"
    echo "    Time: ${ELAPSED}s"
    
    rm predict_hackathon_iter${iteration}.py
    cd ..
}

# Arrays to store timing results
declare -a baseline_times
declare -a triangle_times
declare -a full_times

# CONFIG 1: BASELINE (--no_kernels)
echo -e "${CYAN}[2/4] Running BASELINE (--no_kernels)...${NC}"
run_warmup "baseline" '"--no_kernels"'
echo ""

for i in $(seq 1 ${NUM_ITERATIONS}); do
    run_benchmark_iteration "baseline" ${i} '"--no_kernels"'
    baseline_times+=($(cat "${RESULTS_DIR}/baseline_iter${i}/time.txt"))
done
echo ""

# CONFIG 2: TRIANGLE ONLY
# We need to temporarily modify the code to disable attention kernel
echo -e "${CYAN}[3/4] Running TRIANGLE ONLY (attention kernel disabled)...${NC}"
echo "  Temporarily patching code to disable attention kernel..."

# Create a patch to force use_kernels=False for AttentionPairBias
ATTENTION_FILE="src/boltz/model/layers/attention.py"
cp "${ATTENTION_FILE}" "${ATTENTION_FILE}.backup"

# Replace the forward call to force use_kernels=False
# Find the line where use_kernels is passed and change it to False
python3 << 'PYTHON_PATCH'
import re

with open('src/boltz/model/layers/attention.py', 'r') as f:
    content = f.read()

# Force use_kernels to always be False in the forward method
# Find: "if use_kernels:" and replace with "if False:  # BENCHMARK: Force disable attention kernel"
content = re.sub(
    r'(\s+)if use_kernels:',
    r'\1if False:  # BENCHMARK: Force disable attention kernel',
    content
)

with open('src/boltz/model/layers/attention.py', 'w') as f:
    f.write(content)

print("  Attention kernel disabled")
PYTHON_PATCH

run_warmup "triangle_only" '""'
echo ""

for i in $(seq 1 ${NUM_ITERATIONS}); do
    run_benchmark_iteration "triangle_only" ${i} '""'
    triangle_times+=($(cat "${RESULTS_DIR}/triangle_only_iter${i}/time.txt"))
done

# Restore original file
mv "${ATTENTION_FILE}.backup" "${ATTENTION_FILE}"
echo "  Code restored"
echo ""

# CONFIG 3: FULL OPTIMIZED (all kernels)
echo -e "${CYAN}[4/4] Running FULL OPTIMIZED (all kernels)...${NC}"
run_warmup "full_optimized" '""'
echo ""

for i in $(seq 1 ${NUM_ITERATIONS}); do
    run_benchmark_iteration "full_optimized" ${i} '""'
    full_times+=($(cat "${RESULTS_DIR}/full_optimized_iter${i}/time.txt"))
done
echo ""

# Statistical analysis
echo -e "${BLUE}Computing breakdown analysis...${NC}"

cat > "${RESULTS_DIR}/timing_data.txt" << EOF
baseline: ${baseline_times[@]}
triangle_only: ${triangle_times[@]}
full_optimized: ${full_times[@]}
EOF

python3 << 'PYTHON_ANALYSIS'
import statistics
import json
import math

with open('benchmark_breakdown/timing_data.txt', 'r') as f:
    lines = f.readlines()
    baseline_times = [float(x) for x in lines[0].split(': ')[1].split()]
    triangle_times = [float(x) for x in lines[1].split(': ')[1].split()]
    full_times = [float(x) for x in lines[2].split(': ')[1].split()]

def compute_stats(times):
    return {
        'times': times,
        'mean': statistics.mean(times),
        'std': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times),
        'se': statistics.stdev(times) / math.sqrt(len(times)) if len(times) > 1 else 0
    }

baseline = compute_stats(baseline_times)
triangle = compute_stats(triangle_times)
full = compute_stats(full_times)

# Compute speedups
triangle_speedup = baseline['mean'] / triangle['mean']
full_speedup = baseline['mean'] / full['mean']
attention_contribution = triangle['mean'] / full['mean']

# Time savings
triangle_saved = baseline['mean'] - triangle['mean']
attention_saved = triangle['mean'] - full['mean']
total_saved = baseline['mean'] - full['mean']

results = {
    'baseline': baseline,
    'triangle_only': triangle,
    'full_optimized': full,
    'speedups': {
        'triangle_vs_baseline': triangle_speedup,
        'full_vs_baseline': full_speedup,
        'attention_contribution': attention_contribution
    },
    'time_savings': {
        'triangle_saved_s': triangle_saved,
        'attention_saved_s': attention_saved,
        'total_saved_s': total_saved
    },
    'percent_contributions': {
        'triangle_percent': (triangle_saved / total_saved * 100) if total_saved > 0 else 0,
        'attention_percent': (attention_saved / total_saved * 100) if total_saved > 0 else 0
    }
}

with open('benchmark_breakdown/breakdown_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Analysis complete!")
PYTHON_ANALYSIS

# Cleanup
rm -rf benchmark_temp_breakdown

# Display results
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "KERNEL BREAKDOWN RESULTS"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

python3 << 'PYTHON_DISPLAY'
import json

with open('benchmark_breakdown/breakdown_results.json', 'r') as f:
    results = json.load(f)

b = results['baseline']
t = results['triangle_only']
f = results['full_optimized']
s = results['speedups']
ts = results['time_savings']
pc = results['percent_contributions']

print(f"Configuration: {len(b['times'])} iterations per setup")
print("")
print("TIMING RESULTS:")
print("")
print(f"1. BASELINE (--no_kernels):")
print(f"   Mean:   {b['mean']:.2f}s ± {b['std']:.2f}s")
print(f"   Range:  [{b['min']:.2f}s - {b['max']:.2f}s]")
print("")
print(f"2. TRIANGLE ONLY (attention kernel OFF):")
print(f"   Mean:   {t['mean']:.2f}s ± {t['std']:.2f}s")
print(f"   Range:  [{t['min']:.2f}s - {t['max']:.2f}s]")
print(f"   Speedup vs baseline: {s['triangle_vs_baseline']:.3f}x")
print("")
print(f"3. FULL OPTIMIZED (all kernels ON):")
print(f"   Mean:   {f['mean']:.2f}s ± {f['std']:.2f}s")
print(f"   Range:  [{f['min']:.2f}s - {f['max']:.2f}s]")
print(f"   Speedup vs baseline: {s['full_vs_baseline']:.3f}x")
print("")
print("═══════════════════════════════════════════════════════════════════════════════")
print("BREAKDOWN OF SPEEDUP CONTRIBUTIONS:")
print("═══════════════════════════════════════════════════════════════════════════════")
print("")
print(f"Total speedup:        {s['full_vs_baseline']:.3f}x ({s['full_vs_baseline']*100 - 100:.1f}% faster)")
print(f"Total time saved:     {ts['total_saved_s']:.2f}s")
print("")
print(f"Triangle kernels contribution:")
print(f"  Time saved:     {ts['triangle_saved_s']:.2f}s")
print(f"  Speedup:        {s['triangle_vs_baseline']:.3f}x")
print(f"  % of total gain: {pc['triangle_percent']:.1f}%")
print("")
print(f"AttentionPairBias kernel contribution:")
print(f"  Time saved:     {ts['attention_saved_s']:.2f}s")
print(f"  Speedup:        {s['attention_contribution']:.3f}x")
print(f"  % of total gain: {pc['attention_percent']:.1f}%")
print("")
print("═══════════════════════════════════════════════════════════════════════════════")
print("")

if pc['attention_percent'] > 40:
    print(f"✅ AttentionPairBias kernel provides MAJOR contribution ({pc['attention_percent']:.1f}%)")
elif pc['attention_percent'] > 20:
    print(f"✅ AttentionPairBias kernel provides SIGNIFICANT contribution ({pc['attention_percent']:.1f}%)")
elif pc['attention_percent'] > 10:
    print(f"⚠️  AttentionPairBias kernel provides MODERATE contribution ({pc['attention_percent']:.1f}%)")
else:
    print(f"⚠️  AttentionPairBias kernel provides MINOR contribution ({pc['attention_percent']:.1f}%)")
    print("   Triangle kernels are doing most of the work")

PYTHON_DISPLAY

echo ""
echo "Detailed results saved to: ${RESULTS_DIR}/breakdown_results.json"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
