from litellm import completion
import os
import time
from tqdm import tqdm
import argparse
from openai import OpenAI
from vllm import LLM, SamplingParams


def generator_vllm(model, prompt, max_tokens, task, model_name):
    
    if model_name == "facebook/cwm" or model_name =="facebook/cwm-pretrain":
        prompt = build_prompt_cwm(prompt, task)
        
    prompts = [prompt]
    sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=max_tokens)
    try:
        outputs = model.generate(prompts, sampling_params)
        output = outputs[0]
        generate_text = output.outputs[0].text
    except:
        generate_text = "Error"
        return True, generate_text

    return False, generate_text

def build_prompt_cwm(message,task):
    prompt_content = message.replace("[THOUGHT]","<think>")
    prompt_content = prompt_content.replace("[/THOUGHT]","</think>")
    if task == "input_prediction":
        format_inst = "Always use the tags [INPUT] and [/INPUT] to enclose the predicted input in your external response. DO NOT generate any code or irrelevant text after your response."
    else:
        format_inst = "Always use the tags [OUTPUT] and [/OUTPUT] to enclose the predicted output in your external response. DO NOT generate any code or irrelevant text after your response."
    SYSTEM_PROMPT = (
    "You are a helpful AI assistant. You always reason before responding, "
    "using the following format:\n"
    "<think>\n"
    "your internal reasoning\n"
    "</think>\n"
    "your external response"
    )
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n</s>\n"
        f"<|user|>\n{prompt_content}\n {format_inst} </s>\n"
        f"<|assistant|>\n"
    )

def openrouter_generator(model, prompt, max_new_tokens, max_retries=5, enable_reasoning=False):
    openrouter_key = os.getenv("OPEN_ROUTER_KEY")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key,
    )

    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            if enable_reasoning:
                if "deepseek" in model:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        extra_body={
                            "reasoning": {
                                "enabled":True
                            }
                        }
                    )
                elif "gpt-5" in model:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        extra_body={
                            "reasoning": {
                                "effort": "high"
                            }
                        }
                    )
                else:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        extra_body={
                            "reasoning": {
                                "effort": "high"
                            }
                        }
                    )
                # print(completion.choices[0].message.reasoning)
            else:
                if "gemini" in model:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        extra_body={
                            "reasoning": {
                                "effort": "low"
                            }
                        }
                    )
                elif "claude" in model:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        extra_body={
                            "reasoning": {
                                "enabled":False
                            }
                        }
                    )
                else:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
            # Defensive checks
            if (
                completion is None
                or not hasattr(completion, "choices")
                or not completion.choices
                or completion.choices[0].message is None
                or not completion.choices[0].message.content
                or not completion.choices[0].message.content.strip()
            ):
                print(f"[openrouter_generator] Empty response on attempt {attempt}, retrying...")
                continue

            return False, completion.choices[0].message.content

        except Exception as e:
            last_exception = e
            print(f"[openrouter_generator] Exception on attempt {attempt}: {e}")

    print("[openrouter_generator] Failed after max retries")
    if last_exception:
        print(f"Last exception: {last_exception}")

    return True, ""

def llm_inference(model, task, max_tokens, enable_reasoning=False):
    prompt_root = f"../prompts/{task}"
    model_name = model.split("/")[-1]
    response_root = f"../results/{task}/{model_name}"
    if enable_reasoning:
        response_root = f"../results/{task}/{model_name}-reasoning"
    else:
        response_root = f"../results/{task}/{model_name}"
        
    if model == "deepseek-ai/deepseek-coder-33b-instruct":
        vllm_model = LLM(
            model=model,
            max_model_len=35000,
            download_dir="/u/cliu5/cache_dir", 
            tensor_parallel_size=4
        )
    if model == "facebook/cwm" or model == "facebook/cwm-pretrain":
        vllm_model = LLM(
        model=model,
        dtype="bfloat16",
        tensor_parallel_size=2, 
        max_model_len=32768, 
        download_dir = "/projects/bdsz/cliu5/checkpoints-new",
        gpu_memory_utilization=0.9 # optional but helpful
    )
        
    for difficulty in os.listdir(prompt_root):
        ## difficulty: difficult or medium
        prompt_folder = os.path.join(prompt_root, difficulty)
        
        print(f"Prompting {model} on {difficulty} problems")
        for filename in tqdm(os.listdir(prompt_folder)):
            problem_index, _  = os.path.splitext(filename)
            
            file_path = os.path.join(prompt_folder, filename)
            
            write_folder = os.path.join(response_root, difficulty)
            if not os.path.exists(write_folder):
                os.makedirs(write_folder)
            response_path = os.path.join(write_folder, f"{problem_index}.txt")
            if os.path.exists(response_path):
                continue
            
            with open(file_path, 'r') as f:
                prompt = f.read()
                
            if model == "deepseek-ai/deepseek-coder-33b-instruct" or model == "facebook/cwm" or model == "facebook/cwm-pretrain":    
                err_flag, response = generator_vllm(vllm_model, prompt, max_tokens, task, model)
            err_flag, response = openrouter_generator(model, prompt, max_new_tokens=max_tokens)
            
            if not err_flag:
                with open(response_path, 'w') as f:
                    if response:
                        f.write(response)
                    else:
                        print("Error in problem ", problem_index)
                        f.write("Error")
            else:
                print("Error in problem ", problem_index)
                with open(response_path, 'w') as f:
                    f.write("Error")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--enable_reasoning", action='store_true')
    
    args = parser.parse_args()
    model = args.model
    task = args. task
    max_tokens = args.max_tokens
    enable_reasoning = args.enable_reasoning
    llm_inference(model, task, max_tokens, enable_reasoning)