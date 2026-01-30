#!/bin/bash
TRIPLES=(
    "../../results/validations/gpt-4.1_output_metadata.json|../../results/validations/gpt-4.1_output.csv|output|../../results/summary/gpt-4.1_output_prediction.jsonl"
    "../../results/validations/deepseek-v3_output_metadata.json|../../results/validations/deepseek-v3_output.csv|output|../../results/summary/deepseek-chat-v3-0324_output_prediction.jsonl"
    "../results/validations/gpt-5-mini_input_metadata.json|../results/validations/gpt-5-mini_input.csv|input|../results/summary/gpt-5-mini_input_prediction.jsonl"
    "../../results/validations/deepseek-v3.2_output_metadata.json|../../results/validations/deepseek-v3.2_output.csv|output|../../results/summary/deepseek-v3.2_output_prediction.jsonl"
    "../../results/validations/gemini-3-pro-preview_output_metadata.json|../../results/validations/gemini-3-pro-preview_output.csv|output|../../results/summary/gemini-3-pro-preview_output_prediction.jsonl"
    "../results/validations/gemini-3-pro-preview-reasoning_output_metadata.json|../results/validations/gemini-3-pro-preview-reasoning_output.csv|output|../results/summary/gemini-3-pro-preview-reasoning_output_prediction.jsonl"
    "../results/validations/cwm_output_metadata.json|../results/validations/cwm_output.csv|output|../results/summary/cwm_output_prediction.jsonl"
    "../../results/validations/claude-haiku-4.5-reasoning_output_metadata.json|../../results/validations/claude-haiku-4.5-reasoning_output.csv|output|../../results/summary/claude-haiku-4.5-reasoning_output_prediction.jsonl"
    "../../results/validations/claude-haiku-4.5_output_metadata.json|../../results/validations/claude-haiku-4.5_output.csv|output|../../results/summary/claude-haiku-4.5_output_prediction.jsonl"
    "../results/validations/gpt-5-mini-reasoning_input_metadata.json|../results/validations/gpt-5-mini-reasoning_input.csv|input|../results/summary/gpt-5-mini-reasoning_input_prediction.jsonl"
    "../results/validations/cwm-pretrain_output_metadata.json|../results/validations/cwm-pretrain_output.csv|output|../results/summary/cwm-pretrain_output_prediction.jsonl"
    "../../results/validations/deepseek-v3.2-reasoning_output_metadata.json|../../results/validations/deepseek-v3.2-reasoning_output.csv|output|../../results/summary/deepseek-v3.2-reasoning_output_prediction.jsonl"
)


PYTHON_SCRIPT="./validators/validate_non_swe_bench_results.py"



for triple in "${TRIPLES[@]}"; do
    IFS='|' read -r metadata output task path <<< "$triple"

    echo "Running: python $PYTHON_SCRIPT --summary $path --output_path $output --task $task --metadata_path $metadata"
    python "$PYTHON_SCRIPT" --summary "$path" --output_path "$output" --task "$task" --metadata_path "$metadata"
done