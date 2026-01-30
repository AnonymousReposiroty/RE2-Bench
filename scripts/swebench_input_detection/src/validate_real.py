from utils import read_io, read_executable_code, read_main_code, find_enclosing_class, get_io_id, get_init_params
import os
def is_array_string(v):
    """Check if a value is a string representation of a numpy array."""
    if not isinstance(v, str):
        return False
    return (v.startswith("[") and v.endswith("]")) or v.startswith("array(")


def convert_array_string(v):
    """Convert array string to np.array() call."""
    if v.startswith("array("):
        # Extract the content inside array(...)
        inner = v[6:-1]  # Remove 'array(' and ')'
        # Clean up whitespace and newlines for single-line output
        inner = ' '.join(inner.split())
        return f"np.array({inner})"
    else:
        return f"string2Array({v!r})"


def dict_to_kv_string(d, source="NA"):
    if source == "NA":
        return ', '.join(f"{k} = {v!r}" for k, v in d.items())
    elif source == "sklearn":
        param_list = []
        for k, v in d.items():
            if is_array_string(v):
                param_list.append(f"{k} = {convert_array_string(v)}")
            else:
                param_list.append(f"{k} = {v!r}")
        return ', '.join(param_list)

def loadHelperFunctions(source):
    root_helper = "/home/changshu/RE2-Bench/scripts/swebench_input_detection/src/helpers/"
    if source == "sklearn":
        path_helper = os.path.join(root_helper,"sklearn.py")
        with open(path_helper, "r") as f:
            return f.read()
    return ""
        

def dict_to_kv_string_input(d, key):
    return ', '.join(f"{k} = pred_input['{key}']['{k}']" for k, _ in d.items())

def rename_test_file(problem_id):
    ## remove special characters so that it can be imported
    return problem_id.replace("@@", "__").replace(".", "_").replace("-", "_")
    

def create_call(io_data, enclosed, source="NA"):
    input_data = io_data["input"]
    entry_point = io_data["entry"]
    args = dict_to_kv_string(input_data['args'], source)
    kwargs = dict_to_kv_string(input_data['kwargs'], source)
    
    args_pred = dict_to_kv_string_input(input_data['args'], "args")
    kwargs_pred = dict_to_kv_string_input(input_data['kwargs'], "kwargs")
    
    if kwargs:
        if args:
            combined_args = f"({args}, {kwargs})"
            combined_args_pred = f"({args_pred}, {kwargs_pred})"
        else:
            combined_args = f"({kwargs})"
            combined_args_pred = f"({kwargs_pred})"
    else:
        combined_args = f"({args})"
        combined_args_pred = f"({args_pred})"

    if enclosed:
        return f"assert obj_ins.{entry_point}{combined_args}==obj_ins_pred.{entry_point}{combined_args_pred}, 'Prediction failed!'"
    else:
        return f"assert {entry_point}{combined_args}=={entry_point}{combined_args_pred}, 'Prediction failed!'"

def create_init(io_data, enclosed_class, source, init_params):
    """
    Create initialization code for the ground-truth object.
    Separates __init__ parameters from other attributes.
    """
    input_data = io_data["input"]
    self_data = input_data['self']

    # Handle the case when self_data is a string (direct object representation)
    if isinstance(self_data, str):
        lines = [f"obj_ins = {self_data}"]
        return lines

    # Separate init params from other attributes
    init_dict = {k: v for k, v in self_data.items() if k in init_params}
    attr_dict = {k: v for k, v in self_data.items() if k not in init_params}

    init_str = dict_to_kv_string(init_dict, source)

    lines = [f"obj_ins = {enclosed_class}({init_str})"]

    # Add attribute assignments for non-init params
    for k, v in attr_dict.items():
        if source == "sklearn" and is_array_string(v):
            lines.append(f"obj_ins.{k} = {convert_array_string(v)}")
        else:
            lines.append(f"obj_ins.{k} = {v!r}")

    return lines

def create_init_pred(io_data, enclosed_class, init_params):
    """
    Create initialization code for the predicted object.
    Separates __init__ parameters from other attributes.
    """
    input_data = io_data["input"]
    self_data = input_data['self']

    # Handle the case when self_data is a string (direct object representation)
    if isinstance(self_data, str):
        lines = [f"obj_ins_pred = pred_input['self']"]
        return lines

    # Separate init params from other attributes
    init_keys = [k for k in self_data.keys() if k in init_params]
    attr_keys = [k for k in self_data.keys() if k not in init_params]

    init_str = ', '.join(f"{k} = pred_input['self']['{k}']" for k in init_keys)

    lines = [f"obj_ins_pred = {enclosed_class}({init_str})"]

    # Add attribute assignments for non-init params
    for k in attr_keys:
        lines.append(f"obj_ins_pred.{k} = pred_input['self']['{k}']")

    return lines


def get_source(problem_id):
    if "scikit-learn" in problem_id:
        return "sklearn"
    else:
        return "NA"
    
def rewrite_py_file(problem_id, io_id):
    source = get_source(problem_id)
    wr_dir = f"/home/changshu/RE2-Bench/scripts/swebench_input_detection/playground/{rename_test_file(problem_id)}.py"
    ## read io from ground-truth
    test_func_signature = "def test_input(pred_input):"
    io_data = read_io(problem_id, "Real", io_id)
    entry_point = io_data["entry"]

    original_code = read_main_code(problem_id)
    executable_code = read_executable_code(problem_id)

    enclosing_class = find_enclosing_class(original_code, entry_point)


    if enclosing_class:
        # Get __init__ parameters from the class definition
        init_params = get_init_params(original_code, enclosing_class.name)

        init_lines = create_init(io_data, enclosing_class.name, source, init_params)
        init_pred_lines = create_init_pred(io_data, enclosing_class.name, init_params)
        call_snippet = create_call(io_data, True, source)

        # Combine all init lines with proper indentation
        all_init_lines = init_lines + init_pred_lines + [call_snippet]
        init_body = '\n    '.join(all_init_lines)
        combined_code = f"{executable_code}\n\n{test_func_signature}\n    {init_body}"
    else:
        call_snippet = create_call(io_data, False, source)
        combined_code = f"{executable_code}\n\n{test_func_signature}\n    {call_snippet}"

    helper_functions = loadHelperFunctions(source)
    if helper_functions:
        combined_code = f"{helper_functions}\n\n{combined_code}"

    with open(wr_dir, "w") as f:
        f.write(combined_code)

def process_simple_cases():
    io_data = get_io_id()
    simple_id_path = "swebench-input-detection/src/simple_cases.txt"
    with open(simple_id_path, "r") as f:
        simple_cases = f.read().splitlines()
    for problem_id in simple_cases:
        io_id = io_data[problem_id]
        rewrite_py_file(problem_id, io_id)

def process_real():
    io_data = get_io_id()
    real_id_path = "/home/changshu/RE2-Bench/scripts/swebench_input_detection/src/missing_cases.txt"
    with open(real_id_path, "r") as f:
        real_cases = f.read().splitlines()
    for problem_id in real_cases:
        io_id = io_data[problem_id]
        rewrite_py_file(problem_id, io_id)

if __name__ == "__main__":
    process_real()