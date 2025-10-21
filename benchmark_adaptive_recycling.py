#!/usr/bin/env python3
"""
Benchmark script for Adaptive Recycling optimization.

This script runs predictions with and without adaptive recycling,
measures timing differences, and generates a comparison report.

Usage on AWS:
    # Run baseline (without optimization)
    python benchmark_adaptive_recycling.py \
        --mode baseline \
        --input-jsonl hackathon_data/datasets/abag_public/abag_public.jsonl \
        --msa-dir hackathon_data/datasets/abag_public/msa/ \
        --output-dir ./benchmark_baseline \
        --num-samples 5

    # Run optimized (with adaptive recycling)
    python benchmark_adaptive_recycling.py \
        --mode optimized \
        --input-jsonl hackathon_data/datasets/abag_public/abag_public.jsonl \
        --msa-dir hackathon_data/datasets/abag_public/msa/ \
        --output-dir ./benchmark_optimized \
        --num-samples 5

    # Generate comparison report
    python benchmark_adaptive_recycling.py \
        --mode compare \
        --baseline-dir ./benchmark_baseline \
        --optimized-dir ./benchmark_optimized
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import csv


def read_jsonl(file_path: Path) -> List[Dict]:
    """Read JSONL file and return list of entries."""
    entries = []
    with open(file_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def enable_adaptive_recycling(enable: bool = True):
    """Enable or disable adaptive recycling in boltz2.py."""
    boltz2_path = Path("src/boltz/model/models/boltz2.py")
    
    if not boltz2_path.exists():
        print(f"❌ Error: {boltz2_path} not found!")
        return False
    
    with open(boltz2_path, 'r') as f:
        content = f.read()
    
    import re
    pattern = r'(adaptive_recycling: bool = )(True|False),'
    
    if not re.search(pattern, content):
        print(f"❌ Error: Could not find adaptive_recycling parameter")
        return False
    
    new_value = "True" if enable else "False"
    new_content = re.sub(pattern, rf'\g<1>{new_value},', content)
    
    with open(boltz2_path, 'w') as f:
        f.write(new_content)
    
    status = "ENABLED" if enable else "DISABLED"
    print(f"✅ Adaptive recycling {status}")
    return True


def extract_timing_from_logs(log_file: Path) -> Dict[str, float]:
    """Extract timing information from prediction logs."""
    timings = {
        'recycling_time': None,
        'recycling_steps_actual': None,
        'recycling_steps_max': None,
        'forward_time': None,
        'converged_at_step': None,
        'convergence_mse': None,
    }
    
    if not log_file.exists():
        return timings
    
    with open(log_file) as f:
        content = f.read()
    
    # Parse timing logs
    import re
    
    # [Timing] Recycling completed in 2.34s (2/4 steps)
    match = re.search(r'\[Timing\] Recycling completed in ([\d.]+)s \((\d+)/(\d+) steps\)', content)
    if match:
        timings['recycling_time'] = float(match.group(1))
        timings['recycling_steps_actual'] = int(match.group(2))
        timings['recycling_steps_max'] = int(match.group(3))
    
    # [Adaptive Recycling] Converged at step 2/3 with MSE=0.008234
    match = re.search(r'\[Adaptive Recycling\] Converged at step (\d+)/\d+ with MSE=([\d.]+)', content)
    if match:
        timings['converged_at_step'] = int(match.group(1))
        timings['convergence_mse'] = float(match.group(2))
    
    # [Timing] Total forward pass completed in 5.67s
    match = re.search(r'\[Timing\] Total forward pass completed in ([\d.]+)s', content)
    if match:
        timings['forward_time'] = float(match.group(1))
    
    return timings


def run_benchmark(mode: str, input_jsonl: Path, msa_dir: Path, output_dir: Path, 
                  num_samples: int = 5, result_folder: Path = None):
    """Run benchmark predictions."""
    
    print("=" * 80)
    print(f"Running benchmark: {mode.upper()} mode")
    print("=" * 80)
    
    # Setup
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_dir = output_dir / "submission"
    intermediate_dir = output_dir / "intermediate"
    
    if result_folder is None:
        result_folder = output_dir / "results"
    result_folder.mkdir(parents=True, exist_ok=True)
    
    # Configure adaptive recycling
    if mode == "baseline":
        enable_adaptive_recycling(False)
    elif mode == "optimized":
        enable_adaptive_recycling(True)
    else:
        print(f"❌ Unknown mode: {mode}")
        return None
    
    # Read and limit samples
    all_entries = read_jsonl(input_jsonl)
    entries = all_entries[:num_samples]
    
    print(f"📊 Running predictions on {len(entries)} samples...")
    print(f"   Input: {input_jsonl}")
    print(f"   MSA dir: {msa_dir}")
    print(f"   Output: {output_dir}")
    print()
    
    # Create temporary JSONL with selected samples
    temp_jsonl = output_dir / "samples.jsonl"
    with open(temp_jsonl, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    # Run predictions
    start_time = time.time()
    
    cmd = [
        "python", "hackathon/predict_hackathon.py",
        "--input-jsonl", str(temp_jsonl),
        "--msa-dir", str(msa_dir),
        "--submission-dir", str(submission_dir),
        "--intermediate-dir", str(intermediate_dir),
        "--result-folder", str(result_folder)
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    # Capture output
    log_file = output_dir / "prediction.log"
    with open(log_file, 'w') as log:
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in process.stdout:
                print(line, end='')
                log.write(line)
            
            process.wait()
            
            if process.returncode != 0:
                print(f"❌ Command failed with return code {process.returncode}")
                return None
                
        except Exception as e:
            print(f"❌ Error running predictions: {e}")
            return None
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print()
    print("=" * 80)
    print(f"✅ Benchmark complete!")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Avg per sample: {total_time/len(entries):.2f}s")
    print("=" * 80)
    
    # Extract timing data
    timings = extract_timing_from_logs(log_file)
    
    # Save benchmark results
    results = {
        'mode': mode,
        'num_samples': len(entries),
        'total_time': total_time,
        'avg_time_per_sample': total_time / len(entries),
        'timings': timings,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_file = output_dir / "benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 Results saved to {results_file}")
    
    return results


def compare_results(baseline_dir: Path, optimized_dir: Path):
    """Compare baseline and optimized results."""
    
    print("=" * 80)
    print("BENCHMARK COMPARISON REPORT")
    print("=" * 80)
    print()
    
    # Load results
    baseline_file = baseline_dir / "benchmark_results.json"
    optimized_file = optimized_dir / "benchmark_results.json"
    
    if not baseline_file.exists():
        print(f"❌ Baseline results not found: {baseline_file}")
        return
    
    if not optimized_file.exists():
        print(f"❌ Optimized results not found: {optimized_file}")
        return
    
    with open(baseline_file) as f:
        baseline = json.load(f)
    
    with open(optimized_file) as f:
        optimized = json.load(f)
    
    # Calculate improvements
    total_time_improvement = (baseline['total_time'] - optimized['total_time']) / baseline['total_time'] * 100
    avg_time_improvement = (baseline['avg_time_per_sample'] - optimized['avg_time_per_sample']) / baseline['avg_time_per_sample'] * 100
    
    # Print comparison
    print("📊 OVERALL PERFORMANCE")
    print("-" * 80)
    print(f"{'Metric':<30} {'Baseline':<15} {'Optimized':<15} {'Improvement':<15}")
    print("-" * 80)
    print(f"{'Total Time':<30} {baseline['total_time']:>10.2f}s    {optimized['total_time']:>10.2f}s    {total_time_improvement:>10.1f}%")
    print(f"{'Avg Time/Sample':<30} {baseline['avg_time_per_sample']:>10.2f}s    {optimized['avg_time_per_sample']:>10.2f}s    {avg_time_improvement:>10.1f}%")
    print(f"{'Num Samples':<30} {baseline['num_samples']:>10}       {optimized['num_samples']:>10}       {'N/A':>10}")
    print()
    
    # Detailed timing breakdown
    baseline_timings = baseline.get('timings', {})
    optimized_timings = optimized.get('timings', {})
    
    print("⚡ ADAPTIVE RECYCLING DETAILS")
    print("-" * 80)
    
    if optimized_timings.get('recycling_time') and baseline_timings.get('recycling_time'):
        recycling_improvement = (baseline_timings['recycling_time'] - optimized_timings['recycling_time']) / baseline_timings['recycling_time'] * 100
        print(f"{'Recycling Time':<30} {baseline_timings['recycling_time']:>10.2f}s    {optimized_timings['recycling_time']:>10.2f}s    {recycling_improvement:>10.1f}%")
    
    if optimized_timings.get('forward_time') and baseline_timings.get('forward_time'):
        forward_improvement = (baseline_timings['forward_time'] - optimized_timings['forward_time']) / baseline_timings['forward_time'] * 100
        print(f"{'Forward Pass Time':<30} {baseline_timings['forward_time']:>10.2f}s    {optimized_timings['forward_time']:>10.2f}s    {forward_improvement:>10.1f}%")
    
    if optimized_timings.get('recycling_steps_actual') and optimized_timings.get('recycling_steps_max'):
        steps_saved = optimized_timings['recycling_steps_max'] - optimized_timings['recycling_steps_actual']
        steps_percent = steps_saved / optimized_timings['recycling_steps_max'] * 100
        print(f"{'Recycling Steps':<30} {baseline_timings.get('recycling_steps_max', 'N/A'):>10}       {optimized_timings['recycling_steps_actual']}/{optimized_timings['recycling_steps_max']:<10}  -{steps_percent:>9.1f}%")
    
    if optimized_timings.get('converged_at_step'):
        print(f"{'Converged at Step':<30} {'N/A':>10}       {optimized_timings['converged_at_step']:>10}       {'N/A':>10}")
    
    if optimized_timings.get('convergence_mse'):
        print(f"{'Convergence MSE':<30} {'N/A':>10}       {optimized_timings['convergence_mse']:>10.6f}     {'N/A':>10}")
    
    print()
    
    # Summary
    print("📈 SUMMARY")
    print("-" * 80)
    if total_time_improvement > 0:
        print(f"✅ Adaptive Recycling provides {total_time_improvement:.1f}% speedup")
        print(f"✅ Average time per sample reduced by {avg_time_improvement:.1f}%")
        if optimized_timings.get('recycling_steps_actual') and optimized_timings.get('recycling_steps_max'):
            efficiency = optimized_timings['recycling_steps_actual'] / optimized_timings['recycling_steps_max'] * 100
            print(f"✅ Recycling efficiency: {100-efficiency:.1f}% steps saved")
    else:
        print(f"⚠️  No significant improvement detected")
    print()
    
    # Save comparison report
    report = {
        'baseline': baseline,
        'optimized': optimized,
        'improvements': {
            'total_time_percent': total_time_improvement,
            'avg_time_percent': avg_time_improvement,
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    report_file = Path("benchmark_comparison_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Also save as CSV for easy viewing
    csv_file = Path("benchmark_comparison_report.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Baseline', 'Optimized', 'Improvement (%)'])
        writer.writerow(['Total Time (s)', f"{baseline['total_time']:.2f}", f"{optimized['total_time']:.2f}", f"{total_time_improvement:.1f}"])
        writer.writerow(['Avg Time/Sample (s)', f"{baseline['avg_time_per_sample']:.2f}", f"{optimized['avg_time_per_sample']:.2f}", f"{avg_time_improvement:.1f}"])
        if optimized_timings.get('recycling_time') and baseline_timings.get('recycling_time'):
            recycling_improvement = (baseline_timings['recycling_time'] - optimized_timings['recycling_time']) / baseline_timings['recycling_time'] * 100
            writer.writerow(['Recycling Time (s)', f"{baseline_timings['recycling_time']:.2f}", f"{optimized_timings['recycling_time']:.2f}", f"{recycling_improvement:.1f}"])
    
    print(f"📄 Detailed report saved to {report_file}")
    print(f"📄 CSV report saved to {csv_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Adaptive Recycling optimization for Boltz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--mode",
        choices=["baseline", "optimized", "compare"],
        required=True,
        help="Benchmark mode: baseline, optimized, or compare"
    )
    
    # For baseline/optimized modes
    parser.add_argument("--input-jsonl", type=Path, help="Input JSONL file")
    parser.add_argument("--msa-dir", type=Path, help="MSA directory")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to test (default: 5)")
    parser.add_argument("--result-folder", type=Path, help="Result folder for evaluation")
    
    # For compare mode
    parser.add_argument("--baseline-dir", type=Path, help="Baseline results directory")
    parser.add_argument("--optimized-dir", type=Path, help="Optimized results directory")
    
    args = parser.parse_args()
    
    if args.mode in ["baseline", "optimized"]:
        if not args.input_jsonl or not args.msa_dir or not args.output_dir:
            parser.error(f"{args.mode} mode requires --input-jsonl, --msa-dir, and --output-dir")
        
        run_benchmark(
            mode=args.mode,
            input_jsonl=args.input_jsonl,
            msa_dir=args.msa_dir,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            result_folder=args.result_folder
        )
    
    elif args.mode == "compare":
        if not args.baseline_dir or not args.optimized_dir:
            parser.error("compare mode requires --baseline-dir and --optimized-dir")
        
        compare_results(args.baseline_dir, args.optimized_dir)


if __name__ == "__main__":
    main()
