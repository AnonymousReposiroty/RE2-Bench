#!/bin/bash

task=$1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$task" = "input_prediction" ]; then
    "$SCRIPT_DIR/validate_input_prediction.sh" "${@:2}"
elif [ "$task" = "output_prediction" ]; then
    "$SCRIPT_DIR/validate_output_prediction.sh" "${@:2}"
elif [ "$task" = "input_prediction_nohint" ]; then
    "$SCRIPT_DIR/validate_input_predcition_nohint.sh" "${@:2}"
elif [ "$task" = "input_prediction_nocot" ]; then
    "$SCRIPT_DIR/validate_input_predcition_nocot.sh" "${@:2}"
elif [ "$task" = "output_prediction_nohint" ]; then
    "$SCRIPT_DIR/validate_output_predcition_nohint.sh" "${@:2}"
elif [ "$task" = "output_prediction_nocot" ]; then
    "$SCRIPT_DIR/validate_output_predcition_nocot.sh" "${@:2}"
elif [ "$task" = "loop_prediction" ]; then
    "$SCRIPT_DIR/validate_loop_predcition.sh" "${@:2}"
elif [ "$task" = "branch_prediction" ]; then
    "$SCRIPT_DIR/validate_branch_predcition.sh" "${@:2}"
else
    echo "Unknown task: $task"
    echo "Usage: $0 <input_prediction|output_prediction> [args...]"
    exit 1
fi