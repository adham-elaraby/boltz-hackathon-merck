#!/bin/bash
# Diagnostic script to check what went wrong with the breakdown benchmark

echo "Checking benchmark_breakdown status..."
echo ""

# Check if backup exists
if [ -f "src/boltz/model/layers/attention.py.backup" ]; then
    echo "✓ Backup file exists - restoring..."
    mv src/boltz/model/layers/attention.py.backup src/boltz/model/layers/attention.py
    echo "  File restored"
else
    echo "✗ No backup file found"
fi

# Check log file
if [ -f "benchmark_breakdown/triangle_only_iter1/log.txt" ]; then
    echo ""
    echo "Last 30 lines of triangle_only_iter1 log:"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    tail -n 30 benchmark_breakdown/triangle_only_iter1/log.txt
    echo "═══════════════════════════════════════════════════════════════════════════════"
else
    echo "✗ No log file found at benchmark_breakdown/triangle_only_iter1/log.txt"
fi

# Check if attention.py has the right syntax
echo ""
echo "Checking attention.py syntax..."
python3 -m py_compile src/boltz/model/layers/attention.py 2>&1
if [ $? -eq 0 ]; then
    echo "✓ attention.py syntax is valid"
else
    echo "✗ attention.py has syntax errors"
fi

echo ""
echo "Done. You can now re-run: ./benchmark_kernels_breakdown.sh"
