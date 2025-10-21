"""
Benchmark script to measure actual Boltz inference speed improvement with NVIDIA kernels.
Compares performance with and without cuEquivariance kernels on real data.
"""

import subprocess
import time
import json
import sys
from pathlib import Path
import tempfile
import shutil

def run_boltz_prediction(use_kernels: bool, sample_id: str, output_dir: Path, num_runs: int = 1):
    """Run Boltz prediction and measure time.
    
    Args:
        use_kernels: Whether to use NVIDIA cuEquivariance kernels
        sample_id: ID of the sample to predict
        output_dir: Directory to save outputs
        num_runs: Number of times to run for averaging
        
    Returns:
        Average time in seconds
    """
    total_time = 0.0
    
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}...", end=" ", flush=True)
        
        # Prepare command
        cmd = [
            "python", "-m", "boltz.main", "predict",
            str(sample_id),
            "--out_dir", str(output_dir),
            "--override",
            "--recycling_steps", "3",
            "--num_diffn_timesteps", "50",  # Reduced for faster testing
            "--step_scale", "2.0"
        ]
        
        # Add --no_kernels flag if we want to disable kernels
        if not use_kernels:
            cmd.append("--no_kernels")
        
        # Run and measure time
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=Path.cwd()
            )
            elapsed = time.time() - start_time
            total_time += elapsed
            print(f"{elapsed:.2f}s")
            
        except subprocess.CalledProcessError as e:
            print(f"FAILED!")
            print(f"Error: {e.stderr}")
            return None
    
    return total_time / num_runs


def create_test_input():
    """Create a small test input file for quick benchmarking."""
    test_input = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": ["A"],
                    "sequence": ["MKFLKFSLLTAVLLSVVFAFSSCGDDDDTYPYDVPDYAGYPYDVPDYA"],
                    "msa": None
                }
            }
        ]
    }
    
    test_file = Path("benchmark_test_input.json")
    with open(test_file, "w") as f:
        json.dump(test_input, f, indent=2)
    
    return test_file


def benchmark_with_dataset(dataset_name: str = "abag_public", num_samples: int = 1, num_runs: int = 1):
    """Benchmark using existing dataset files.
    
    Args:
        dataset_name: Name of the dataset (abag_public or asos_public)
        num_samples: Number of samples to test
        num_runs: Number of runs per configuration for averaging
    """
    print("=" * 80)
    print(f"BOLTZ INFERENCE BENCHMARK - {dataset_name.upper()}")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Samples: {num_samples}")
    print(f"  Runs per config: {num_runs}")
    print(f"  Recycling steps: 3")
    print(f"  Diffusion timesteps: 50 (reduced for faster testing)")
    print("=" * 80)
    
    # Check if dataset exists
    dataset_path = Path(f"hackathon_data/datasets/{dataset_name}/{dataset_name}.jsonl")
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return
    
    # Read sample IDs
    sample_ids = []
    with open(dataset_path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line)
            sample_ids.append(data["id"])
    
    print(f"\nTesting with samples: {', '.join(sample_ids)}")
    
    # Create temporary output directories
    output_dir_no_kernels = Path(tempfile.mkdtemp(prefix="boltz_benchmark_no_kernels_"))
    output_dir_with_kernels = Path(tempfile.mkdtemp(prefix="boltz_benchmark_with_kernels_"))
    
    try:
        results = {}
        
        # Test WITHOUT kernels (baseline)
        print("\n" + "=" * 80)
        print("PHASE 1: Testing WITHOUT NVIDIA Kernels (Baseline)")
        print("=" * 80)
        
        times_no_kernels = []
        for sample_id in sample_ids:
            print(f"\nProcessing {sample_id}...")
            avg_time = run_boltz_prediction(
                use_kernels=False,
                sample_id=f"hackathon_data/datasets/{dataset_name}/{sample_id}",
                output_dir=output_dir_no_kernels,
                num_runs=num_runs
            )
            
            if avg_time is None:
                print("Benchmark failed!")
                return
            
            times_no_kernels.append(avg_time)
            print(f"  Average: {avg_time:.2f}s")
        
        avg_time_no_kernels = sum(times_no_kernels) / len(times_no_kernels)
        results["no_kernels"] = {
            "times": times_no_kernels,
            "average": avg_time_no_kernels
        }
        
        # Test WITH kernels (optimized)
        print("\n" + "=" * 80)
        print("PHASE 2: Testing WITH NVIDIA Kernels (Optimized)")
        print("=" * 80)
        
        times_with_kernels = []
        for sample_id in sample_ids:
            print(f"\nProcessing {sample_id}...")
            avg_time = run_boltz_prediction(
                use_kernels=True,
                sample_id=f"hackathon_data/datasets/{dataset_name}/{sample_id}",
                output_dir=output_dir_with_kernels,
                num_runs=num_runs
            )
            
            if avg_time is None:
                print("Benchmark failed!")
                return
            
            times_with_kernels.append(avg_time)
            print(f"  Average: {avg_time:.2f}s")
        
        avg_time_with_kernels = sum(times_with_kernels) / len(times_with_kernels)
        results["with_kernels"] = {
            "times": times_with_kernels,
            "average": avg_time_with_kernels
        }
        
        # Calculate speedup
        speedup = avg_time_no_kernels / avg_time_with_kernels
        time_saved = avg_time_no_kernels - avg_time_with_kernels
        percent_faster = (speedup - 1) * 100
        
        # Print results
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"\nAverage inference time per sample:")
        print(f"  WITHOUT kernels (baseline): {avg_time_no_kernels:.2f}s")
        print(f"  WITH kernels (optimized):   {avg_time_with_kernels:.2f}s")
        print(f"\nPerformance improvement:")
        print(f"  Time saved:     {time_saved:.2f}s per sample")
        print(f"  Speedup:        {speedup:.2f}x")
        print(f"  Percent faster: {percent_faster:.1f}%")
        
        if speedup > 1.0:
            print(f"\n✓ SUCCESS: Kernels provide {percent_faster:.1f}% speedup!")
        else:
            print(f"\n⚠ WARNING: No speedup observed. This might be due to:")
            print("  - Short sequences (kernels optimize longer sequences)")
            print("  - Reduced diffusion steps (50 vs 200)")
            print("  - Cold cache / warmup needed")
        
        # Per-sample breakdown
        print(f"\nPer-sample breakdown:")
        print(f"{'Sample':<20} {'No Kernels':<15} {'With Kernels':<15} {'Speedup':<10}")
        print("-" * 60)
        for i, sample_id in enumerate(sample_ids):
            sample_speedup = times_no_kernels[i] / times_with_kernels[i]
            print(f"{sample_id:<20} {times_no_kernels[i]:>10.2f}s    {times_with_kernels[i]:>10.2f}s    {sample_speedup:>7.2f}x")
        
        # Save results to file
        results_file = Path("benchmark_results.json")
        with open(results_file, "w") as f:
            json.dump({
                "dataset": dataset_name,
                "num_samples": num_samples,
                "num_runs": num_runs,
                "sample_ids": sample_ids,
                "results": results,
                "speedup": speedup,
                "percent_faster": percent_faster
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        print("=" * 80)
        
    finally:
        # Cleanup
        print("\nCleaning up temporary directories...")
        shutil.rmtree(output_dir_no_kernels, ignore_errors=True)
        shutil.rmtree(output_dir_with_kernels, ignore_errors=True)
        print("Done!")


def quick_test():
    """Quick test with a simple protein sequence."""
    print("=" * 80)
    print("QUICK BENCHMARK TEST")
    print("=" * 80)
    print("Testing with a simple protein sequence...")
    
    # Create test input
    test_file = create_test_input()
    
    try:
        output_dir_no_kernels = Path(tempfile.mkdtemp(prefix="boltz_quick_no_kernels_"))
        output_dir_with_kernels = Path(tempfile.mkdtemp(prefix="boltz_quick_with_kernels_"))
        
        # Test without kernels
        print("\nTest 1: WITHOUT kernels")
        time_no_kernels = run_boltz_prediction(
            use_kernels=False,
            sample_id=str(test_file),
            output_dir=output_dir_no_kernels,
            num_runs=1
        )
        
        # Test with kernels
        print("\nTest 2: WITH kernels")
        time_with_kernels = run_boltz_prediction(
            use_kernels=True,
            sample_id=str(test_file),
            output_dir=output_dir_with_kernels,
            num_runs=1
        )
        
        if time_no_kernels and time_with_kernels:
            speedup = time_no_kernels / time_with_kernels
            print("\n" + "=" * 80)
            print("QUICK TEST RESULTS")
            print("=" * 80)
            print(f"WITHOUT kernels: {time_no_kernels:.2f}s")
            print(f"WITH kernels:    {time_with_kernels:.2f}s")
            print(f"Speedup:         {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)")
        
        # Cleanup
        shutil.rmtree(output_dir_no_kernels, ignore_errors=True)
        shutil.rmtree(output_dir_with_kernels, ignore_errors=True)
        test_file.unlink()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Boltz with/without NVIDIA kernels")
    parser.add_argument("--dataset", type=str, default="abag_public",
                       choices=["abag_public", "asos_public"],
                       help="Dataset to use for benchmarking")
    parser.add_argument("--num_samples", type=int, default=1,
                       help="Number of samples to test")
    parser.add_argument("--num_runs", type=int, default=1,
                       help="Number of runs per configuration for averaging")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test with synthetic data")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        benchmark_with_dataset(
            dataset_name=args.dataset,
            num_samples=args.num_samples,
            num_runs=args.num_runs
        )
