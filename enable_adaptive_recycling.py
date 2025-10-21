#!/usr/bin/env python3
"""
Quick setup script to enable Adaptive Recycling optimization for Boltz.

This script modifies the boltz2.py file to enable adaptive recycling by default.
Use this for quick testing/benchmarking during the hackathon.

Usage:
    python enable_adaptive_recycling.py
    
To disable:
    python enable_adaptive_recycling.py --disable
"""

import sys
from pathlib import Path
import argparse


def enable_adaptive_recycling(disable=False):
    """Enable or disable adaptive recycling in boltz2.py"""
    
    boltz2_path = Path("src/boltz/model/models/boltz2.py")
    
    if not boltz2_path.exists():
        print(f"❌ Error: {boltz2_path} not found!")
        print("Make sure you run this script from the boltz-hackathon-template-dsc root directory.")
        return False
    
    # Read the file
    with open(boltz2_path, 'r') as f:
        content = f.read()
    
    if disable:
        # Disable: set to False
        new_value = "False"
        action = "Disabling"
    else:
        # Enable: set to True
        new_value = "True"
        action = "Enabling"
    
    # Replace the adaptive_recycling parameter
    # Look for the line: adaptive_recycling: bool = False,
    # or: adaptive_recycling: bool = True,
    
    import re
    pattern = r'(adaptive_recycling: bool = )(True|False),'
    
    if not re.search(pattern, content):
        print(f"❌ Error: Could not find adaptive_recycling parameter in {boltz2_path}")
        print("The file may have been modified. Please check manually.")
        return False
    
    # Replace
    new_content = re.sub(pattern, rf'\g<1>{new_value},', content)
    
    # Write back
    with open(boltz2_path, 'w') as f:
        f.write(new_content)
    
    print(f"✅ {action} adaptive recycling in {boltz2_path}")
    print(f"   adaptive_recycling: bool = {new_value}")
    
    return True


def show_status():
    """Show current adaptive recycling status"""
    
    boltz2_path = Path("src/boltz/model/models/boltz2.py")
    
    if not boltz2_path.exists():
        print(f"❌ Error: {boltz2_path} not found!")
        return
    
    with open(boltz2_path, 'r') as f:
        content = f.read()
    
    import re
    match = re.search(r'adaptive_recycling: bool = (True|False)', content)
    
    if match:
        status = match.group(1)
        if status == "True":
            print("✅ Adaptive recycling is currently ENABLED")
        else:
            print("❌ Adaptive recycling is currently DISABLED")
    else:
        print("⚠️  Could not determine adaptive recycling status")


def main():
    parser = argparse.ArgumentParser(
        description="Enable/disable Adaptive Recycling optimization in Boltz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enable adaptive recycling (default)
  python enable_adaptive_recycling.py
  
  # Disable adaptive recycling
  python enable_adaptive_recycling.py --disable
  
  # Check current status
  python enable_adaptive_recycling.py --status

After enabling, run predictions normally:
  python hackathon/predict_hackathon.py --input-json examples/specs/example_protein_complex.json ...
  
Check the output for timing improvements:
  [Timing] Recycling completed in X.XXs (N/M steps)
  [Adaptive Recycling] Converged at step N/M with MSE=0.XXXXXX
        """
    )
    
    parser.add_argument(
        "--disable",
        action="store_true",
        help="Disable adaptive recycling (set to False)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current status without making changes"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Boltz Adaptive Recycling - Quick Setup")
    print("=" * 80)
    print()
    
    if args.status:
        show_status()
        print()
        print("To enable:  python enable_adaptive_recycling.py")
        print("To disable: python enable_adaptive_recycling.py --disable")
    else:
        if enable_adaptive_recycling(disable=args.disable):
            print()
            print("Next steps:")
            print("-" * 80)
            print("1. Run predictions with your normal command")
            print("2. Watch for [Timing] and [Adaptive Recycling] log messages")
            print("3. Compare timing with baseline (without optimization)")
            print()
            print("Example:")
            print("  python hackathon/predict_hackathon.py \\")
            print("      --input-json examples/specs/example_protein_complex.json \\")
            print("      --msa-dir hackathon_data/datasets/abag_public/msa \\")
            print("      --submission-dir submission \\")
            print("      --intermediate-dir intermediate")
            print()
            print("Expected output:")
            print("  [Timing] Recycling completed in 2.34s (2/4 steps)")
            print("  [Adaptive Recycling] Converged at step 2/3 with MSE=0.008234")
            print("  [Timing] Total forward pass completed in 5.67s")
        else:
            print()
            print("❌ Setup failed. Please check error messages above.")
            return 1
    
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
