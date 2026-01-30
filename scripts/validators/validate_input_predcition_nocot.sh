#!/bin/bash

TRIPLES=(
    "../results/validations/gemini-3-pro-preview-reasoning_input_prediction_wocot_metadata.json|../results/validations/gemini-3-pro-preview-reasoning_input_prediction_wocot.csv|input|../results/summary/gemini-3-pro-preview-reasoning_input_prediction_wocot.jsonl"
    "../results/validations/claude-haiku-4.5-reasoning_input_prediction_wocot_metadata.json|../results/validations/claude-haiku-4.5-reasoning_input_prediction_wocot.csv|input|../results/summary/claude-haiku-4.5-reasoning_input_prediction_wocot.jsonl"
    "../results/validations/gpt-5-mini-reasoning_input_prediction_wocot_metadata.json|../results/validations/gpt-5-mini-reasoning_input_prediction_wocot.csv|input|../results/summary/gpt-5-mini-reasoning_input_prediction_wocot.jsonl"
    "../results/validations/deepseek-v3.2-reasoning_input_prediction_wocot_metadata.json|../results/validations/deepseek-v3.2-reasoning_input_prediction_wocot.csv|input|../results/summary/deepseek-v3.2-reasoning_input_prediction_wocot.jsonl"
    "../results/validations/cwm_input_prediction_wocot_metadata.json|../results/validations/cwm_input_prediction_wocot.csv|input|../results/summary/cwm_input_prediction_wocot.jsonl"
)

PYTHON_SCRIPT="./validators/validate_non_swe_bench_results.py"

for triple in "${TRIPLES[@]}"; do
    IFS='|' read -r metadata output task path <<< "$triple"

    echo "Running: python $PYTHON_SCRIPT --summary $path --output_path $output --task $task --metadata_path $metadata"
    python "$PYTHON_SCRIPT" --summary "$path" --output_path "$output" --task "$task" --metadata_path "$metadata"
done