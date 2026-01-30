import json
from .utils import read_io, get_io_id

import os
import re
import subprocess
import sys
from typing import Tuple, Dict, Any
from .converter import SklearnConverter, SympyConverter
import traceback

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

def rename_test_file(problem_id):
    ## remove special characters so that it can be imported
    return problem_id.replace("@@", "__").replace(".", "_").replace("-", "_")
    

def extract_input_test(model_id, problem_id):
    # jsonl_path = f"../results/summary/input_prediction/{model_id}_input_prediction.jsonl"
    jsonl_path = f"../results/summary/{model_id}_input_prediction.jsonl"
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("problem_id") == problem_id:
                result = obj
                break
    return result['prediction']


    

def create_run_file(input_predicted, problem_id):
    renamed_id = rename_test_file(problem_id)
    path_code = f"swebench_input_detection/playground/{renamed_id}.py"
    run_code = open(path_code, "r").read()
    
    
    full_code = f"{run_code}"
    test_file_path = f"swebench_input_detection/src/test_case.py"
    with open(test_file_path, "w") as f:
        f.write(full_code)



# def has_pred_failed_assertion(file_path: str, timeout: float = 30.0) -> int:
#     if not os.path.isfile(file_path):
#         raise FileNotFoundError(f"No such file: {file_path}")
#     try:
#         proc = subprocess.run(
#             [sys.executable, file_path],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             stdin=subprocess.DEVNULL,
#             timeout=timeout,
#             text=True,
#         )
#         combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
#         combined_clean = ANSI_ESCAPE_RE.sub("", combined)

#         if "AssertionError: Prediction failed!" in combined_clean:
#             return 1
#         if proc.returncode != 0:
#             return 1
#         return 0
#     except subprocess.TimeoutExpired as e:
#         # Treat timeouts as errors
#         return 1

def run_test_file(input_predicted):
    ## import the test_case.py here
    try:
        from .test_case import test_input
    except ImportError:
        return 1
    try:
        test_input(input_predicted)
        return 0
    except Exception as e:
        print("*"*10)
        print(e)
        traceback.print_exc()
        print("*"*10)
        return 1

def test_run(model_id, problem_id):
    try:
        test_input = extract_input_test(model_id, problem_id)
    except Exception as e:
        print(f"Error extracting input test: {e}")
        return 1
    if problem_id.startswith("scikit-learn") or problem_id.startswith("django") or problem_id.startswith("astropy") or problem_id.startswith("pydata_xarray"):
        # Convert string representation of lists to actual lists
        test_input = SklearnConverter().convert(test_input)
    elif problem_id.startswith("sympy__sympy"):
        test_input = SympyConverter().convert(test_input)
    try:
        create_run_file(test_input, problem_id)
    except Exception as e:
        print(f"Error creating run file: {e}")
        return 1

    # result = has_pred_failed_assertion(test_file_path)
    result = run_test_file(test_input)
    return result

if __name__ == "__main__":
    print("called")
    model_id = sys.argv[1]
    problem_id= sys.argv[2]
    res = test_run(model_id, problem_id)
    if res == 0:
        print(f"Test passed for {problem_id}")
    else:
        print(f"Test failed for {problem_id}")
    sys.exit(res)
    