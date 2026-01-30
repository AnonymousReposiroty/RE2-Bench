import json
import os
import ast
from typing import Optional

# Get the directory containing this script for relative path resolution
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

def read_io(index, source, io_id):
    ## return the input, output, and the entry point.
    input_ouput_root = os.path.join(_PROJECT_ROOT, "dataset", "re2-bench", "input-output", source)
    if source == "Swebench":
        modified_filename = index.split("@@")[0] + "@@" + index.split("@@")[1].rstrip(".py") + "." + index.split("@@")[2] + ".jsonl"
    elif source == "cruxeval":
        modified_filename = index.replace("sample", "cruxeval") + ".jsonl"
    elif source == "HumanEval":
        modified_filename = index + ".jsonl"
    elif source == "Classeval":
        modified_filename = index.split("@")[-1].replace(".", "@") + ".jsonl"
    elif source == "Avatar":
        modified_filename = index + ".jsonl"
    else :
        modified_filename = index + ".jsonl"
    file_path = os.path.join(input_ouput_root, modified_filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
    else:
        ## sample an input-output pair
        with open(file_path, 'r') as f:
            for idx, line in enumerate(f):
                if idx == io_id:
                    io_info = json.loads(line)
                    if source == "Avatar":
                        input_info = io_info["values"]["input"]
                        output_info = io_info["values"]["output"]
                        return {"input": input_info, "output": output_info, "entry": ""}
                    else:
                        input_info = io_info["values"]["inputs"]
                        output_info = io_info["values"]["return"]
                        entry_point = io_info["name"]
                        return {"input": input_info, "output": output_info, "entry": entry_point}

def read_executable_code(index):
    code_path = f"/home/changshu/RE2-Bench/scripts/swebench_input_detection/tmp/{index}.py"
    executable_code = open(code_path, "r").read()
    return executable_code

def read_main_code(index):
    code_path = os.path.join(_PROJECT_ROOT, "dataset", "re2-bench", "code", f"{index}.py")
    source_code = open(code_path, "r").read()
    return source_code

def find_enclosing_class(source: str, method_name: str) -> Optional[ast.ClassDef]:
    """
    Parse `source` and return the ast.ClassDef that encloses a function
    named `method_name`. If the function is not inside any class, return None.
    """
    tree = ast.parse(source)
    result: Optional[ast.ClassDef] = None
    class_stack: list[ast.ClassDef] = []

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef):
            class_stack.append(node)
            self.generic_visit(node)
            class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            nonlocal result
            if node.name == method_name and class_stack:
                # innermost enclosing class of this method
                result = class_stack[-1]
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            # treat async functions the same way
            self.visit_FunctionDef(node)  # type: ignore[arg-type]

    Visitor().visit(tree)
    return result


def get_init_params(source: str, class_name: str):
    """
    Parse `source` and return the list of parameter names for the __init__ method
    of the class with `class_name`. Excludes 'self'.
    """
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                    params = []
                    for arg in item.args.args:
                        if arg.arg != 'self':
                            params.append(arg.arg)
                    for arg in item.args.kwonlyargs:
                        params.append(arg.arg)
                    return params
    return []


def get_io_id():
    overall_data = {}
    json_path = os.path.join(_PROJECT_ROOT, "dataset", "re2-bench", "sampled_problems.json")
    with open(json_path, "r") as f:
        data = json.load(f)
        for k in data["difficult"]:
            if data["difficult"][k]["benchmark"] == "Swebench" or data["difficult"][k]["benchmark"] == "Real":
                overall_data[k] = data["difficult"][k]["input-output"]
        for k in data["easy"]:
            if data["easy"][k]["benchmark"] == "Swebench" or data["easy"][k]["benchmark"] == "Real":
                overall_data[k] = data["easy"][k]["input-output"]
    return overall_data
