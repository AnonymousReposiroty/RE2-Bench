#!/bin/bash

# Validation script for RE2-Bench results
# Usage depends on task type:
#   input_prediction/output_prediction: validate_results.sh --task <task> --model <model_id>
#   loop_prediction: validate_results.sh --task loop_prediction --model <model_id>
#   branch_prediction: validate_results.sh --task branch_prediction --model <model_id>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
TASK=""
MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate task parameter
if [[ -z "$TASK" ]]; then
    echo "Error: --task parameter is required"
    echo "Valid tasks: input_prediction, output_prediction, loop_prediction, branch_prediction"
    exit 1
fi

# Validate model parameter
if [[ -z "$MODEL" ]]; then
    echo "Error: --model parameter is required"
    exit 1
fi

# Extract model name from path (e.g., "openai/gpt-5-mini" -> "gpt-5-mini")
MODEL="${MODEL##*/}"

case "$TASK" in
    input_prediction|output_prediction|input_prediction_wohint|input_prediction_wocot|output_prediction_wohint|output_prediction_wocot)
        # Determine task type for the Python script (input or output)
        if [[ "$TASK" == "input_prediction" || "$TASK" == "input_prediction_wohint" || "$TASK" == "input_prediction_wocot" ]]; then
            PYTHON_TASK="input"
        else
            PYTHON_TASK="output"
        fi

        # Infer paths from model and task
        SUMMARY="../results/summary/${MODEL}_${TASK}.jsonl"
        OUTPUT_PATH="../results/validations/${MODEL}_${TASK}.csv"
        METADATA_PATH="../results/validations/${MODEL}_${TASK}_metadata.json"

        python3 "$SCRIPT_DIR/validate_reasoning_results.py" \
            --summary "$SUMMARY" \
            --output_path "$OUTPUT_PATH" \
            --task "$PYTHON_TASK" \
            --metadata_path "$METADATA_PATH"
        ;;

    loop_prediction)
        python3 "$SCRIPT_DIR/validate_loop.py" --model "$MODEL"
        ;;

    branch_prediction)
        python3 "$SCRIPT_DIR/validate_branch.py" --model "$MODEL"
        ;;

    *)
        echo "Error: Invalid task '$TASK'"
        echo "Valid tasks: input_prediction, output_prediction, loop_prediction, branch_prediction"
        exit 1
        ;;
esac
