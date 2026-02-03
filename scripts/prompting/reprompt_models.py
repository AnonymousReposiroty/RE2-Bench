from litellm import completion
import os
import time
from tqdm import tqdm
import argparse
from openai import OpenAI


def litellm_generator(model, prompt, max_new_tokens):
    err_flag = False
    inference_not_done = True
    output = ""
    
    if model == "gpt-4o-mini":
        temperature = 1
    else:
        temperature = 0.0
    while inference_not_done:
        try:
            output = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=1.0
            )
            inference_not_done = False
        except Exception as e:
            print(e)
            if 'maximum context length is' in str(e):
                inference_not_done = False
                err_flag = True
                output = ''
            else:
                time.sleep(3)
    return err_flag, output.choices[0]['message']['content']


def deepseek_generator(prompt, max_new_tokens):
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
    try:
        response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=False,
        max_tokens=max_new_tokens,
        temperature=0.0
        )
        return False, response.choices[0].message.content
    except Exception as e:
        print(e)
        return True, ""


def openrouter_generator(model, prompt, max_new_tokens):
    try:
        openrouter_key = os.getenv("OPEN_ROUTER_KEY")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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",
        )
        completion = client.chat.completions.create(
            model = model,
            messages=[
                {
                "role": "user",
                "content": prompt
                }
            ]
        )
        return False, completion.choices[0].message.content
    except Exception as e:
        print(e)
        return True, ""    
    

def get_incomplete_response(response_folder, tag):
    incomplete_files = []
    for filename in os.listdir(response_folder):
        file_path = os.path.join(response_folder, filename)
        with open(file_path, 'r') as f:
            content = f.read()
            if content.strip() == "Error" or content.strip() == "" or tag not in content:
                incomplete_files.append(filename)
    return incomplete_files

def llm_inference(model, task, max_tokens):
    prompt_root = f"../prompts/{task}"
    model_name = model.split("/")[-1]
    response_root = f"../results/{task}/{model_name}"
    for difficulty in os.listdir(prompt_root):
        ## difficulty: difficult or medium
        prompt_folder = os.path.join(prompt_root, difficulty)
        if task == "output_prediction":
            tag = "[/OUTPUT]"
        else:
            tag = "[/INPUT]"
        incomplete_ids = get_incomplete_response(os.path.join(response_root, difficulty), tag)
        print(f"Prompting {model} on {difficulty} problems")
        for filename in tqdm(incomplete_ids):
            problem_index, _  = os.path.splitext(filename)
            
            file_path = os.path.join(prompt_folder, filename)
            
            write_folder = os.path.join(response_root, difficulty)
            if not os.path.exists(write_folder):
                os.makedirs(write_folder)
            response_path = os.path.join(write_folder, f"{problem_index}.txt")
            # if os.path.exists(response_path):
            #     continue
            
            with open(file_path, 'r') as f:
                prompt = f.read()
            if model == "deepseek/deepseek-reasoner":
                err_flag, response = openrouter_generator("deepseek/deepseek-r1-0528", prompt, max_new_tokens=max_tokens)
            elif model == "gemini/gemini-2.5-pro":
                err_flag, response = openrouter_generator("google/gemini-2.5-pro", prompt, max_new_tokens=max_tokens)
            else:
                err_flag, response = litellm_generator(model, prompt, max_new_tokens=max_tokens)
            
            if not err_flag:
                with open(response_path, 'w') as f:
                    if response:
                        print("Resolved problem ", problem_index)
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
    
    args = parser.parse_args()
    model = args.model
    task = args. task
    max_tokens = args.max_tokens
    llm_inference(model, task, max_tokens)