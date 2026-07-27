#!/bin/bash

# Script to extract types for non-SWE-Bench datasets
# Usage: ./extract_non_swe_bench_types.sh

PYTHON_SCRIPT="./non_swe_bench_type_extractor.py"
BASE_PATH="../../dataset/re2-bench/input-output"

# List of projects to process
PROJECTS=(
    "HumanEval"
    "Classeval"
    "CruxEval"
    "Avatar"
)

echo "Extracting types for non-SWE-Bench datasets..."

for project in "${PROJECTS[@]}"; do
    project_path="$BASE_PATH/$project"
    
    if [ -d "$project_path" ]; then
        echo "Processing project: $project"
        echo "Path: $project_path"
        
        python "$PYTHON_SCRIPT" --project "$project" --path "$project_path"
        
        echo "Completed processing $project"
        echo "---"
    else
        echo "Warning: Directory $project_path does not exist, skipping $project"
    fi
done

echo "Non-SWE-Bench type extraction completed!"
