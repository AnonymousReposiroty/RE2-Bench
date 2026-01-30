from litellm import completion
import os
import time
from tqdm import tqdm
import argparse
from openai import OpenAI


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
                                "effort": "high",
                            }
                        }
                    )
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
    if enable_reasoning:
        response_root = f"../results/{task}/{model_name}-reasoning"
    else:
        response_root = f"../results/{task}/{model_name}"
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
                
 
            err_flag, response = openrouter_generator(model, prompt, max_new_tokens=max_tokens, enable_reasoning=enable_reasoning)
            
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
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--enable_reasoning", action='store_true')
    
    args = parser.parse_args()
    model = args.model
    task = args. task
    max_tokens = args.max_tokens
    enable_reasoning = args.enable_reasoning
    llm_inference(model, task, max_tokens, enable_reasoning)