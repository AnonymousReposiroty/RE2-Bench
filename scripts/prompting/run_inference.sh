MODEL_ID=$1
TASK=$2
REASONING=$3

if [ "$REASONING" = "true" ]; then
    python prompting/prompt_models.py --model $MODEL_ID --task $TASK --enable_reasoning
else
    python prompting/prompt_models.py --model $MODEL_ID --task $TASK
fi
if [ "$TASK" = "input_prediction" ] || [ "$TASK" = "input_prediction_wohint" ] || [ "$TASK" = "input_prediction_wocot" ] || [ "$TASK" = "output_prediction" ] || [ "$TASK" = "output_prediction_wohint" ] || [ "$TASK" = "output_prediction_wocot" ]; then
    python prompting/parse_results.py --model $MODEL_ID --task $TASK
fi