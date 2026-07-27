#!/bin/bash

# Script to extract types for SWE-Bench dataset
# Usage: ./extract_swe_bench_types.sh

PYTHON_SCRIPT="./swe_bench_type_extractors.py"
SWE_BENCH_PATH="../../dataset/re2-bench/input-output/Swebench"

echo "Extracting types for SWE-Bench dataset..."
echo "Path: $SWE_BENCH_PATH"

python "$PYTHON_SCRIPT" --path "$SWE_BENCH_PATH"

echo "SWE-Bench type extraction completed!"
