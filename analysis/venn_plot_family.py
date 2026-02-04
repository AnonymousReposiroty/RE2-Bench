import csv
from matplotlib_venn import venn2
import json
import os 
from matplotlib import pyplot as plt

def get_results(model_name, task):
    csv_path = f"../results/validations/{model_name}_{task}.csv"
    results = {"difficult": [], "easy": []}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_name = row['case_name']
            # Remove .json suffix if present
            if case_name.endswith('.json'):
                case_name = case_name[:-5]
            elif case_name.endswith('.jsonl'):
                case_name = case_name[:-6]
            difficulty = row['difficulty_level']
            if int(row['rs']) == 1:
                results[difficulty].append(case_name)
    return results

def plot_venn(list1, list2, title):
    plt.cla()
    set1 = set(list1)
    set2 = set(list2)

    venn = venn2([set1, set2], set_labels=('', ''))
    for text in venn.subset_labels:
        if text:  # avoid None
            text.set_fontsize(50)
    save_path = f"../analysis/figs/training/{title}.jpg"
    plt.savefig(save_path,dpi=300, bbox_inches='tight', pad_inches=0)
    
    
def main(task):
    result_claude = get_results("claude-haiku-4.5-reasoning", task)
    result_claude_lr = get_results("claude-haiku-4.5", task)
    plot_venn(result_claude["difficult"], result_claude_lr["difficult"], f"claude_{task}_difficult")
    plot_venn(result_claude["easy"], result_claude_lr["easy"], f"claude_{task}_easy")
    
    result_deepseek = get_results("deepseek-v3.2-reasoning", task)
    result_deepseek_lr = get_results("deepseek-v3.2", task)
    plot_venn(result_deepseek["difficult"], result_deepseek_lr["difficult"], f"deepseek_{task}_difficult")
    plot_venn(result_deepseek["easy"], result_deepseek_lr["easy"], f"deepseek_{task}_easy")
    
    result_gemini = get_results("gemini-3-pro-preview-reasoning", task)
    result_gemini_lr = get_results("gemini-3-pro-preview", task)
    plot_venn(result_gemini["difficult"], result_gemini_lr["difficult"], f"gemini_{task}_difficult")
    plot_venn(result_gemini["easy"], result_gemini_lr["easy"], f"gemini_{task}_easy")
    
    result_gpt = get_results("gpt-5-mini-reasoning", task)
    result_gpt_lr = get_results("gpt-5-mini", task)
    plot_venn(result_gpt["difficult"], result_gpt_lr["difficult"], f"gpt_{task}_difficult")
    plot_venn(result_gpt["easy"], result_gpt_lr["easy"], f"gpt_{task}_easy")
    
    result_cwm = get_results("cwm", task)
    result_cwm_pretrain = get_results("cwm-pretrain", task)
    plot_venn(result_cwm["difficult"], result_cwm_pretrain["difficult"], f"cwm_{task}_difficult")
    plot_venn(result_cwm["easy"], result_cwm_pretrain["easy"], f"cwm_{task}_easy")
    
    

if __name__ == "__main__":
    main("output")
    main("input")