import json
import os
def instruction():
    instruction = """You will be given:
1. A Python code snippet wrapped in [PYTHON] ... [/PYTHON]. The code includes branch markers in comments of the form # [STATE]{VARIABLE_NAME}=??[/STATE].
2. A method input wrapped in [INPUT] ... [/INPUT].
Your task is to replace every "??" beween [STATE] and [/STATE] with your prediction of the state of variables associated with LOOPS.

Detailed Instructions:
* Replace ?? with a list.
* You need to predict the states of variables in For loops, While loops, or List Comprehensions.
* If the value of a varibale stays the same through K iterations, repeat its value for K times as its state.
* Determine variable states by tracing the code step by step. Wrap your reasoning in [THOUGHT] ... [/THOUGHT]
* Output the fully annotated code (with ?? replaced) wrapped in [ANSWER] ... [/ANSWER]
* Do not remove, reorder, or add any code lines. 
* Preserve the original line numbers exactly as they appear in the [PYTHON] ... [/PYTHON] block.
"""
    return instruction

def load_icl_template(problem_id):
    root = "./prompts_icl_examples/loop_prediction"
    if problem_id.startswith("HumanEval") or problem_id.startswith("sample"):
        template_path = f"{root}/humaneval_cruxeval.txt"
    elif problem_id.startswith("atcoder") or problem_id.startswith("codeforces"):
        template_path = f"{root}/avatar.txt"
    elif problem_id.startswith("ClassEval"):
        template_path = f"{root}/classeval.txt"
    else:
        template_path = f"{root}/swebench.txt"
    with open(template_path, "r") as f:
        template = f.read()
    return template


def load_input(problem_id, difficulty):
    f_path= f"../dataset/re2-bench/loop/{difficulty}/{problem_id}.json"
    with open(f_path, "r") as f:
        data = f.read()
        data = json.loads(data)
    method_name = data.get("name", None)
    input = data.get("input", None)
    input_text = json.dumps(input, indent=4)
    if method_name:
        input_str = f"The input to the method ```{method_name}``` is: \n {input_text}"
    else:
        input_str = f"The input is: \n {input["raw_input"]}"
    return input_str

def load_annotated_file(problem_id, difficulty):
    f_path= f"../prompts/loop_prediction-intermediate/{difficulty}/{problem_id}.txt"
    with open(f_path, "r") as f:
        lines = f.readlines()
    numbered_lines = [f"{i+1} {line.rstrip()}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)


def create_prompt(problem_id, difficulty):
    inst = instruction()
    icl = load_icl_template(problem_id)
    input = load_input(problem_id, difficulty)
    annotated_code = load_annotated_file(problem_id, difficulty)
    prompt = inst + "\n" + "Please follow the format in the example below:" + "\n\n" +\
        icl + "\n\n" + annotated_code + "\n\n" + input + "\n\n" + 'Complete the loop annotations in the code with variable states.\n'
    
    wr_folder = f"../prompts/loop_prediction/{difficulty}"
    if not os.path.exists(wr_folder):
        os.makedirs(wr_folder)
    wr_path = f"{wr_folder}/{problem_id}.txt"
    with open(wr_path, "w") as f:
        f.write(prompt)

if __name__ == "__main__":
    
    root = "../dataset/re2-bench/loop/"
    
    ## create prompts for difficult problems:
    dir_difficult = f"{root}/difficult"
    for i in os.listdir(path=dir_difficult):
        fname = i.replace(".json","")
        create_prompt(fname, "difficult")
    dir_easy = f"{root}/easy"
    for i in os.listdir(path=dir_easy):
        fname = i.replace(".json","")
        create_prompt(fname, "easy")
        
        