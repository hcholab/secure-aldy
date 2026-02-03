#!/bin/bash
# Monitor the benchmark progress

LOG_FILE="/home/hs2286/aldy/benchmark_10samples.log"

echo "Monitoring benchmark progress..."
echo "================================"
echo ""

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found: $LOG_FILE"
    exit 1
fi

# Show last 30 lines
echo "Last 30 lines of output:"
echo "------------------------"
tail -30 "$LOG_FILE"

echo ""
echo "================================"
echo "Current progress:"
grep -E "Sample [0-9]+/10:" "$LOG_FILE" | tail -1

echo ""
echo "To continue monitoring, run:"
echo "  tail -f $LOG_FILE"
