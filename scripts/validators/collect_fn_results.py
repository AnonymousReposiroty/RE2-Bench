import argparse
import csv
import json
import subprocess
import rich
from json import JSONDecodeError
from typing import List, Any
import os
from collect_fn_reports import add_fn_reports
from collect_scores import add_scores_report
from compare_partially import partial_compare
from detect_false_negative_cases import detect_false_negative
from generate_reports import create_detail_csv_file, ValidationOutputPaths, create_report_json_file
from get_case_info import CaseInfo, get_class_name, get_case_index, read_case_name
from load_original_code import get_code_path
from deserialize import deserialize


def is_complete_correct(partial_rs):
    return 1 if partial_rs == 1 else 0


def get_fn_detector_result(rs, predicted_value, ground_truth_input, ground_truth_output, function_name,
                           case_info: CaseInfo):
    try:
        is_fn = detect_false_negative(predicted_value, ground_truth_input,
                                      function_name, ground_truth_output,
                                      case_info) if rs == 0 else False
    except subprocess.TimeoutExpired as e:
        print(f"{case_info.task}-{case_info.benchmark}-{case_info.case_name}-{case_info.case_index}- timeout")
        is_fn = False
    except subprocess.CalledProcessError as e:
        print(f"{case_info.task}-{case_info.benchmark}-{case_info.case_name}-{case_info.case_index}- error: {e.stderr}")
        is_fn = False
    return is_fn

def get_fn_detector_result_swe(rs, predicted_value, problem_id):
    if rs == 1:
        return False
    predicted_value_json = json.dumps(predicted_value, ensure_ascii=False)
    result = subprocess.run(
                ["python", "-m", "swebench_input_detection.src.run_test_swe", predicted_value_json, problem_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, 
                text=True                 
            )
    # print("STDOUT:", result.stdout)
    if "Test failed for" in result.stdout:
        return False
    else:
        return True

def validate_prediction_per_case(case_info: CaseInfo, ground_truths, predicted_value):
    ground_truth_input = ground_truths['inputs']
    ground_truth_output = ground_truths['outputs']
    task = case_info.task
    benchmark = case_info.benchmark
    if task == 'input' and benchmark != 'Swebench' and benchmark != "Real":
        deserialized_prediction = deserialize(predicted_value)
        deserialized_gt = deserialize(ground_truth_input)
        function_name = ground_truths['function_name']
        count_and_score = {'score': 0.0, 'count': 0}
        partial_compare(deserialized_prediction, deserialized_gt, count_and_score)
        partial_rs = count_and_score['score'] / count_and_score['count']
        rs = is_complete_correct(partial_rs)
        is_fn = get_fn_detector_result(rs, deserialized_prediction, deserialized_gt, ground_truth_output,
                                       function_name,
                                       case_info)
        if is_fn:
            rs = 1
            partial_rs = 1  
        return {'rs': rs, 'partial_rs': partial_rs, 'is_fn': is_fn}
    elif task == 'input' and benchmark == "Real":
        # todo: deserialize - currently we don't deserialize because they have numpy arrays and we handle them in partially comparing
        deserialized_predicted_input = predicted_value
        count_and_score = {'score': 0.0, 'count': 0}
        partial_compare(deserialized_predicted_input, ground_truth_input, count_and_score)
        partial_rs = count_and_score['score'] / count_and_score['count']
        rs = is_complete_correct(partial_rs)
        is_fn = get_fn_detector_result_swe(rs, deserialized_predicted_input, case_info.problem_id)
        if is_fn:
            rs = 1
            partial_rs = 1
        return {'rs': rs, 'partial_rs': partial_rs, 'is_fn': is_fn}
    elif task == 'input' and benchmark == 'Swebench':
        # todo: deserialize - currently we don't deserialize because they have numpy arrays and we handle them in partially comparing
        deserialized_predicted_input = predicted_value
        count_and_score = {'score': 0.0, 'count': 0}
        partial_compare(deserialized_predicted_input, ground_truth_input, count_and_score)
        partial_rs = count_and_score['score'] / count_and_score['count']
        rs = is_complete_correct(partial_rs)
        is_fn = get_fn_detector_result_swe(rs, deserialized_predicted_input, case_info.problem_id)
        if is_fn:
            rs = 1
            partial_rs = 1
        return {'rs': rs, 'partial_rs': partial_rs, 'is_fn': is_fn}
    elif task == 'output':
        deserialized_predicted_output = predicted_value
        if isinstance(predicted_value, dict) and 'output' in predicted_value.keys():
            deserialized_predicted_output = predicted_value['output']
        count_and_score = {'score': 0.0, 'count': 0}
        partial_compare(deserialized_predicted_output, ground_truth_output, count_and_score)
        partial_rs = count_and_score['score'] / count_and_score['count']
        rs = is_complete_correct(partial_rs)
        return {'rs': rs, 'partial_rs': partial_rs}
    return None


def load_result_jsons_model(model_summary_path: str) -> List[Any]:
    results = []
    with open(model_summary_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except Exception as e:
                    print(f"Oh not json parsable {line}")
    return results


def load_ground_truths(benchmark, case_name, index):
    base_path = f'../dataset/re2-bench/input-output/{benchmark}/{case_name}'
    with (open(base_path, "r", encoding="utf-8") as f):
        for i, line in enumerate(f):
            if i == index:
                try:
                    master_json = json.loads(line)
                except JSONDecodeError as e:
                    print(case_name, index)
                values = master_json['values']
                return {'inputs': values['inputs'] if 'inputs' in values.keys() else values['input'],
                        'outputs': values['return'] if 'return' in values.keys() else values['output'],
                        'function_name': master_json['name'] if 'name' in master_json.keys() else None,
                        'class_name': get_class_name(case_name, benchmark)}

    raise IndexError(f"Index {index} is out of range for {base_path}")


def get_code(code_path):
    return open(code_path, "r", encoding="utf-8").read()

def collect(model_summary_path, task, fn_result, model_name):
    fn_results = {}
    model_task_cases_results = load_result_jsons_model(model_summary_path)
    wr_folder = f"/home/changshu/Artifacts/RE2-Bench/false_negative_input_prediction/{model_name}"
    if not os.path.exists(wr_folder):
        os.makedirs(wr_folder)
    for case_result in model_task_cases_results:
        problem_id = case_result['problem_id']
        difficulty_level = case_result['difficulty']
        case_index, benchmark = get_case_index(problem_id, difficulty_level)
        case_name = read_case_name(problem_id, benchmark)
        ground_truths = load_ground_truths(benchmark, case_name, case_index)
        code_path = get_code_path(problem_id, benchmark)
        predicted_value = case_result['prediction']
        case_info = CaseInfo(case_name, case_index, problem_id, benchmark, task, code_path)
        is_fn = fn_result[case_name]
        if is_fn:
            code = get_code(code_path)
            function_name = ground_truths['function_name'] if ground_truths['function_name'] else ""
            gt_input = ground_truths['inputs']
            gt_output = ground_truths['outputs']
            
            reasoning_path = f"../results/input_prediction/{model_name}/{difficulty_level}/{problem_id}.txt"
            llm_response = open(reasoning_path, 'r', encoding='utf-8').read()
            
            report = "<<FUNCTION_NAME>>\n\n"+function_name+"\n\n<<CODE>>\n\n"+code+"\n\n<<GROUND_TRUTH_INPUT>>\n\n"+str(gt_input)+"\n\n<<GROUND_TRUTH_OUTPUT>>\n\n"+str(gt_output)+"\n\n<<LLM_RESPONSE>>\n\n"+llm_response
            
            wr_path = f"{wr_folder}/{problem_id}.txt"
            with open(wr_path, 'w', encoding='utf-8') as f:
                f.write(report)
            # fn_results[problem_id] = {
            #     "code": code,
            #     "ground_truth_input": gt_input,
            #     'predicted_input': predicted_value,
            #     'difficulty': difficulty_level,
            #     'function_name': function_name
            # }
            
    return fn_results
            

def read_csv(csv_path):
    result = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_name = row['case_name']
            is_fn = row['is_fn'].lower() == 'true'
            result[case_name] = is_fn
    return result

def main(summary_path, csv_path, model_name):
    task_type = "input"  # or "output", depending on your needs
    fn_result = read_csv(csv_path)
    fn_results = collect(summary_path, task_type, fn_result, model_name)
    return fn_results
    

if __name__ == "__main__":
    # fn_haiku_reasoning = main("../results/summary/claude-haiku-4.5-reasoning_input_prediction.jsonl",
                    # "../results/validations/claude-haiku-4.5-reasoning_input.csv", "claude-haiku-4.5-reasoning")
    # fn_haiku = main("../results/summary/claude-haiku-4.5_input_prediction.jsonl",
                    # "../results/validations/claude-haiku-4.5_input.csv", "claude-haiku-4.5")
    # fn_cwm = main("../results/summary/cwm_input_prediction.jsonl",
    #                 "../results/validations/cwm_input.csv", "cwm")
    # fn_cwm_pretrain = main("../results/summary/cwm-pretrain_input_prediction.jsonl",
    #                 "../results/validations/cwm-pretrain_input.csv", "cwm-pretrain")
    # fn_deepseek_reasoning = main("../results/summary/deepseek-v3.2-reasoning_input_prediction.jsonl",
    #                 "../results/validations/deepseek-v3.2-reasoning_input.csv", "deepseek-v3.2-reasoning")
    # fn_deepseek = main("../results/summary/deepseek-v3.2_input_prediction.jsonl",
    #                 "../results/validations/deepseek-v3.2_input.csv", "deepseek-v3.2")
    # fn_gemini_reasoning = main("../results/summary/gemini-3-pro-preview-reasoning_input_prediction.jsonl",
    #                 "../results/validations/gemini-3-pro-preview-reasoning_input.csv", "gemini-3-pro-preview-reasoning")
    # fn_gemini = main("../results/summary/gemini-3-pro-preview_input_prediction.jsonl",
    #                 "../results/validations/gemini-3-pro-preview_input.csv", "gemini-3-pro-preview")
    # fn_gpt_reasoning = main("../results/summary/gpt-5-mini-reasoning_input_prediction.jsonl",
    #                 "../results/validations/gpt-5-mini-reasoning_input.csv", "gpt-5-mini-reasoning")
    fn_gpt = main("../results/summary/gpt-5-mini_input_prediction.jsonl",
                    "../results/validations/gpt-5-mini_input.csv","gpt-5-mini")
    # results = {
    #     "Claude-Haiku-4.5": fn_haiku_reasoning,
    #     "Claude-Haiku-4.5(RD)": fn_haiku,
    #     "CWM": fn_cwm,
    #     "CWM-Pretrain": fn_cwm_pretrain,
    #     "Deepseek-v3.2": fn_deepseek_reasoning,
    #     "Deepseek-v3.2(RD)": fn_deepseek,
    #     "Gemini-3-Pro": fn_gemini_reasoning,
    #     "Gemini-3-Pro(LR)": fn_gemini,
    #     "GPT-5-mini": fn_gpt_reasoning,
    #     "GPT-5-mini(RD)": fn_gpt
    # }
    
    # with open("/home/changshu/Artifacts/RE2-Bench/fn_cases_details.json", 'w', encoding='utf-8') as f:
    #     json.dump(results, f, indent=4, ensure_ascii=False)