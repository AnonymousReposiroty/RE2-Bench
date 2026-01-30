#!/bin/bash
TRIPLES_INPUT=(
    "../results/validations/gpt-5-mini_input_metadata.json|../results/validations/gpt-5-mini_input.csv|input|../results/summary/gpt-5-mini_input_prediction.jsonl"
    "../results/validations/deepseek-v3.2_input_metadata.json|../results/validations/deepseek-v3.2_input.csv|input|../results/summary/deepseek-v3.2_input_prediction.jsonl"
    "../results/validations/gemini-3-pro-preview_input_metadata.json|../results/validations/gemini-3-pro-preview_input.csv|input|../results/summary/gemini-3-pro-preview_input_prediction.jsonl"
    "../results/validations/gemini-3-pro-preview-reasoning_input_metadata.json|../results/validations/gemini-3-pro-preview-reasoning_input.csv|input|../results/summary/gemini-3-pro-preview-reasoning_input_prediction.jsonl"
    "../results/validations/cwm_input_metadata.json|../results/validations/cwm_input.csv|input|../results/summary/cwm_input_prediction.jsonl"
    "../results/validations/claude-haiku-4.5-reasoning_input_metadata.json|../results/validations/claude-haiku-4.5-reasoning_input.csv|input|../results/summary/claude-haiku-4.5-reasoning_input_prediction.jsonl"
    "../results/validations/claude-haiku-4.5_input_metadata.json|../results/validations/claude-haiku-4.5_input.csv|input|../results/summary/claude-haiku-4.5_input_prediction.jsonl"
    "../results/validations/gpt-5-mini-reasoning_input_metadata.json|../results/validations/gpt-5-mini-reasoning_input.csv|input|../results/summary/gpt-5-mini-reasoning_input_prediction.jsonl"
    "../results/validations/cwm-pretrain_input_metadata.json|../results/validations/cwm-pretrain_input.csv|input|../results/summary/cwm-pretrain_input_prediction.jsonl"
    "../results/validations/deepseek-v3.2-reasoning_input_metadata.json|../results/validations/deepseek-v3.2-reasoning_input.csv|input|../results/summary/deepseek-v3.2-reasoning_input_prediction.jsonl"
)

PYTHON_SCRIPT="./validators/validate_non_swe_bench_results.py"

for triple in "${TRIPLES_INPUT[@]}"; do
    IFS='|' read -r metadata output task path <<< "$triple"

    echo "Running: python $PYTHON_SCRIPT --summary $path --output_path $output --task $task --metadata_path $metadata"
    python "$PYTHON_SCRIPT" --summary "$path" --output_path "$output" --task "$task" --metadata_path "$metadata"
done