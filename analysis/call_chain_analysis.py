
import ast
import json
import os
import csv
import matplotlib.pyplot as plt

def read_sampling_results():
    samples = []
    sample_path = "../dataset/re2-bench/sampled_problems.json"
    with open(sample_path, 'r') as f:
        sampling_results = json.load(f)
        for i in sampling_results:
            for k in sampling_results[i]:
                samples.append(
                    {
                        "difficulty": i,
                        "id": k,
                        "source": sampling_results[i][k]['benchmark'],
                        "io_id": sampling_results[i][k]['input-output']
                    }
                )
    return samples


def read_code_path(index, source):
    if source in ["Swebench", "Avatar", "HumanEval", "Real"]:
        code_path = f"../dataset/re2-bench/code/{index}.py"
    elif source == "Classeval":
        code_path = f"../dataset/re2-bench/code/{index.split("@")[-1]}.py"
    elif source == "cruxeval":
        code_path = f"../dataset/re2-bench/code/{index.replace("sample", "cruxeval")}.py"
    return code_path

def read_dep_path(index):
    return f"../dataset/re2-bench/dependency/{index}.json"
    
        
def count_functions_in_file(filename):
    with open(filename, "r") as f:
        tree = ast.parse(f.read(), filename=filename) 
    function_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    return max(1,len(function_defs))

def read_io_name(index, source, io_id):
    ## return the input, output, and the entry point.
    input_ouput_root = f"../dataset/re2-bench/input-output/{source}"
    if source == "Swebench":
        modified_filename = index.split("@@")[0] + "@@" + index.split("@@")[1].rstrip(".py") + "." + index.split("@@")[2]
    elif source == "cruxeval":
        modified_filename = index.replace("sample", "cruxeval")
    elif source == "HumanEval":
        modified_filename = index
    elif source == "Classeval":
        modified_filename = index.split("@")[-1].replace(".", "@")
    elif source == "Avatar":
        modified_filename = index
    elif source == "Real":
        modified_filename = index
    return modified_filename

def count_deps_in_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    dependencies = len(data.keys())
    return dependencies
    # return min(dependencies, 50)

def collect_dependencies():
    results = {}
    samples = read_sampling_results()
    for sample in samples:
        index = sample["id"]
        source = sample["source"]
        code_path = read_code_path(index, source)
        io_id = sample["io_id"]
        io_name = read_io_name(index, source, io_id)
        if source in ["Swebench", "Real"]:
            dep_path = read_dep_path(index)
            num_funcs = count_functions_in_file(code_path)
            num_deps = count_deps_in_json(dep_path)
            num_deps = num_deps + num_funcs
        else:
            num_deps = count_functions_in_file(code_path)
        results[io_name] = num_deps
    path = "../analysis/dependencies.json"
    with open(path, 'w') as wr:
        json.dump(results, wr, indent=4)
        
def get_results(model_name, task):
    csv_path = f"../results/validations/{model_name}_{task}.csv"
    results = {"difficult": {}, "easy": {}}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_name = row['case_name']
            # Remove .json suffix if present
            if case_name.endswith('.json'):
                case_name = case_name[:-5]
            elif case_name.endswith('.jsonl'):
                case_name = case_name[:-6]
            results[case_name] = int(row['rs'])
    return results

def load_data(model_id, task):
    succ_deps, fail_deps = [], []
    results = get_results(model_id, task)
    deps_data = json.load(open("../analysis/dependencies.json", 'r'))
    for k in results:
        if k not in deps_data:
            continue
        dep = deps_data[k]
        if results[k]:
            succ_deps.append(dep)
            # if task == "output":
            #     succ_deps.append(dep+2.8 if dep > 1 else dep)
            # else:
            #     succ_deps.append(dep)
        else:
            fail_deps.append(dep)
    return succ_deps, fail_deps

def plot_box(task):
    succ_claude_reasoning, fail_claude_reasoning = load_data("claude-haiku-4.5-reasoning", task)
    succ_claude, fail_claude = load_data("claude-haiku-4.5", task)
    succ_deepseek_reasoning, fail_deepseek_reasoning = load_data("deepseek-v3.2-reasoning", task)
    succ_deepseek, fail_deepseek = load_data("deepseek-v3.2", task)
    succ_gemini_reasoning, fail_gemini_reasoning = load_data("gemini-3-pro-preview-reasoning", task)
    succ_gemini, fail_gemini = load_data("gemini-3-pro-preview", task)
    succ_gpt_reasoning, fail_gpt_reasoning = load_data("gpt-5-mini-reasoning", task)
    succ_gpt, fail_gpt = load_data("gpt-5-mini", task)
    succ_cwm, fail_cwm = load_data("cwm", task)
    succ_cwm_pretrain, fail_cwm_pretrain = load_data("cwm-pretrain", task)
    
    data = [succ_claude_reasoning, fail_claude_reasoning, succ_claude, fail_claude,
            succ_deepseek_reasoning, fail_deepseek_reasoning, succ_deepseek, fail_deepseek,
            succ_gemini_reasoning, fail_gemini_reasoning, succ_gemini, fail_gemini,
            succ_gpt_reasoning, fail_gpt_reasoning, succ_gpt, fail_gpt,
            succ_cwm, fail_cwm, succ_cwm_pretrain, fail_cwm_pretrain]
    plt.figure(figsize=(6, 2))
    bp = plt.boxplot(
        data,
        showmeans=True,
        widths=0.7,
        patch_artist=True,
        meanline=True
        )
    colors = ['lightyellow', 'lightyellow', 'greenyellow', "greenyellow", "aquamarine", "aquamarine", "lightpink", "lightpink", "mistyrose", "mistyrose", "cyan", "cyan", "orange", "orange", "lightblue", "lightblue", "violet", "violet", "lightgray", "lightgray"]
    savepath = f"../analysis/figs/call_chain/call_chain_{task}.jpg"

    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], ['S', 'F', 'S', 'F', 'S', 'F', 'S', 'F', 'S', 'F', 'S', 'F', 'S', 'F', 'S', 'F', 'S', 'F', 'S', 'F'])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0, 50)
    plt.tight_layout()
    plt.savefig(savepath, dpi=500)    
    
if __name__ == "__main__":
    collect_dependencies()
    plot_box("output")
    plot_box("input")