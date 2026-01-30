
<p align="center">
    <a href="./false_negatives_example.md">⚠️ False Negatives in Input Prediction</a>
    <a href="/failure_cases.md">🔥Additional Reasoning Failure Cases</a> 
</p>

## Installation
Please visit OpenRouter (https://openrouter.ai/) and create a `open router key`, then set it in your local environment:
```
export OPEN_ROUTER_KEY="your_key_here"
```

Run the following commands to create the environment:
```
conda create -n re2-bench python=3.9
pip install -r requirements.txt
conda activate re2-bench
```


## Reproducing Results

### Prompting LLMs:

Model_ID: {`anthropic/claude-haiku-4.5`, `deepseek/deepseek-v3.2`, `google/gemini-3-pro-preview`, `openai/gpt-5-mini`, `facebook/cwm-pretrain`, `facebook/cwm`}

TASK: {`input_prediction`, `output_prediction`, `loop_prediction`, `branch_prediction`, `input_prediction_wohint`, `input_prediction_wocot`, `output_predcition_wohint`, `output_prediction_wocot`}

By default, LLMs are evaluted under low reasoning effort/reasoning disabled setting.
Please use `--enable_reasoning` for high reasoning effort/reasoning enabled variants.
```
cd scripts
python prompting/prompt_models.py --model {MODEL_ID} --task {TASK}  {--enable_reasoning}
```

Responses from LLM will be hosted under `results/{TASK}`

### Validation on the results


To validate all generated responses in `results/`, run:

```bash
cd scripts/validators
./validate.sh --{TASK}
```

This command will generate csv files under `results/validations` with the summaries of the performance of LLMs on the specified code reasoning tasks.


### How to evaluate new LLMs
Our pipeline supports all the available LLMs on OpenRouter. Please visite https://openrouter.ai/ to get the correct model id and use it for experiments.