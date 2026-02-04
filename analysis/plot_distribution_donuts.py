import csv
from utils import donuts_distribution_with_label, donuts_distribution_without_label
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 2.0
meta_data_dataset = {
    'high': {
        'Swebench': 93,
        "Real": 157,
        'Avatar': 0,
        'cruxeval': 0,
        'HumanEval': 0,
        'Classeval': 0
    },
    'low': {
        'Swebench': 43,
        "Real": 43,
        'Avatar': 34,
        'cruxeval': 44,
        'HumanEval': 43,
        'Classeval': 43
    },    
}


def count_dataset(items, cat):
    count_swebench = items.count("Swebench")
    count_classeval = items.count("Classeval")
    count_avatar = items.count("Avatar")
    count_cruxeval = items.count("cruxeval")
    count_humaneval = items.count("HumanEval")
    count_real = items.count("Real")
    return [count_swebench, meta_data_dataset[cat]["Swebench"]-count_swebench,
        count_classeval, meta_data_dataset[cat]["Classeval"]- count_classeval,
        count_avatar, meta_data_dataset[cat]["Avatar"] - count_avatar,
        count_cruxeval, meta_data_dataset[cat]["cruxeval"] - count_cruxeval,
        count_humaneval, meta_data_dataset[cat]["HumanEval"] - count_humaneval,
        count_real, meta_data_dataset[cat]["Real"] - count_real]

def load_input_data(model):
    difficulty_instances, easy_instances = [], []
    path = f"../results/validations/{model}_input.csv"
    with open(path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            
            if int(row["rs"]) == 1:
                if row["difficulty_level"] == "difficult":
                    difficulty_instances.append(row["benchmark"])
                else:
                    easy_instances.append(row["benchmark"])
            if int(row["rs"]) == 0 and row["is_fn"] == "True":
                if row["difficulty_level"] == "difficult":
                    difficulty_instances.append(row["benchmark"])
                else:
                    easy_instances.append(row["benchmark"])
    return difficulty_instances, easy_instances

def load_output_data(model):
    difficulty_instances, easy_instances = [], []
    path = f"../results/validations/{model}_output.csv"
    with open(path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            
            if int(row["rs"]) == 1:
                if row["difficulty_level"] == "difficult":
                    difficulty_instances.append(row["benchmark"])
                else:
                    easy_instances.append(row["benchmark"])
    return difficulty_instances, easy_instances

if __name__ == "__main__":
    # models = ["gpt-4-turbo", "gemini/gemini-1.5-pro", "gemini/gemini-2.5-pro", "deepseek/deepseek-reasoner"]
    models = ["claude-haiku-4.5-reasoning", "claude-haiku-4.5","gpt-5-mini-reasoning", "gpt-5-mini", "cwm", "cwm-pretrain", "deepseek-v3.2-reasoning", "deepseek-v3.2", "gemini-3-pro-preview-reasoning", "gemini-3-pro-preview"]
    for model in models:
        difficulty_instances_input, easy_instances_input = load_input_data(model)
        count_high_input = count_dataset(difficulty_instances_input, "high")
        count_low_input = count_dataset(easy_instances_input, "low")
        
        input_results = {
            "strong": count_high_input,
            "weak": count_low_input
        }
        print(input_results)
        save_path_input_with_label = f"./figs/distribution/{model}_input.jpeg"
        save_path_input_wo_label = f"./figs/distribution/{model}_input_wo.jpeg"
        donuts_distribution_with_label(input_results, save_path_input_with_label)
        # donuts_distribution_without_label(input_results, save_path_input_wo_label)
        
        difficulty_instances_output, easy_instances_output = load_output_data(model)
        count_high_output = count_dataset(difficulty_instances_output, "high")
        count_low_output = count_dataset(easy_instances_output, "low")
        
        output_results = {
            "strong": count_high_output,
            "weak": count_low_output
        }
        
        save_path_output_with_label = f"./figs/distribution/{model}_output.jpeg"
        save_path_output_wo_label = f"./figs/distribution/{model}_output_wo.jpeg"
        donuts_distribution_with_label(output_results, save_path_output_with_label)
        # donuts_distribution_without_label(output_results, save_path_output_wo_label)