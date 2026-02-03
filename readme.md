
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
conda create -n re2-bench python=3.12
pip install -r requirements.txt
conda activate re2-bench
```

We provide a Dockerfile to reproduce the results of RE2-bench.
Execute the following to create a docker image and execute the container in interactive mode:
```
docker build -t re2bench .
docker run -it re2bench bash
```

If you are using MacOS with an Apple chip, please consider adding `--platform=linux/amd64` in docker build.


## Reproducing Results

### Prompting LLMs:

RE2-Bench is compatiable with LLMs supposrted by OpenRouter and Vllm.
```
cd scripts
bash prompting/run_inference.sh {MODEL_ID} {TASK} {HIGH_REASONING}
```

`MODEL_ID`: currently supports `anthropic/claude-haiku-4.5`, `deepseek/deepseek-v3.2`, `google/gemini-3-pro-preview`, `openai/gpt-5-mini`, `facebook/cwm-pretrain`, `facebook/cwm`, `openai/o4-mini`, and `google/gemini-2.5-pro`.

`TASK`: currently supports the following four code reasoning tasks: `input_prediction`, `output_prediction`, `loop_prediction`, `branch_prediction`, as well as the ablation varients for the input/output prediction: `input_prediction_wohint`, `input_prediction_wocot`, `output_predcition_wohint`, `output_prediction_wocot`.

`HIGH_REASONING`: `true` or `false`. `true` refers to setting reasoning effort/budget to `high` while `false` means setting it to "None"(if applicable) or "low".

### Validation on the results
To evaluate LLMs' performance, store the response as strings into a jsonl file under `results/summary/`. The naming convention is "{MODEL}_{TASK}.jsonl", e.g., "gpt-5-mini_input_prediction.jsonl".
Each line should follow the following format:
```json
{
    "model": "gpt-5-mini_input_prediction.jsonl", 
    "prediction": {"self": {}, "args": {"newbackend": "agg"}, "kwargs": {}}, 
    "problem_id": "matplotlib@@matplotlib_pyplot.py@@switch_backend_L393", 
    "difficulty": "HC"
}
```
If you inference with RE2-Bench, these files will be automaticlly generated for you.
Then run the following command to get validation results:
```
cd scripts
bash validators/validate_results.sh --task {TASK} --model {MODEL_ID}
```
This script should take a while to run because it requires ruuning test cases. once it finishes, you will find evaluation results under `results/validations`


### How to evaluate new LLMs
The inference component supports LLMs served by [OpenRouter](https://openrouter.ai/) and [vLLM](https://docs.vllm.ai/en/latest/).
Please go to their websites and use correct model IDs.