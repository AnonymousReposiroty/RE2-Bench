import json
import re
import os

def parse_real_problem_name(problem_id):
    """
    Parse Real benchmark problem name like:
    'sympy@@sympy_core_expr.py@@_eval_is_extended_positive_negative_L912'

    Returns (module_path, function_name, line_number)
    """
    parts = problem_id.split('@@')
    if len(parts) != 3:
        return None, None, None

    package, file_path, func_with_line = parts

    # Remove .py extension
    module_path = file_path.replace('.py', '')

    # Handle special prefixes like 'src_' for attrs, 'lib.' for matplotlib
    if module_path.startswith('src_'):
        module_path = module_path[4:]  # Remove 'src_'

    # Handle __init__ specially - it becomes just the parent module
    # e.g., matplotlib___init__ -> matplotlib
    module_path = re.sub(r'___init__$', '', module_path)
    module_path = re.sub(r'___init___', '_', module_path)

    # Replace __ with a placeholder first (for private modules like _nmf, _make)
    module_path = module_path.replace('__', ':::PRIVATE:::')

    # Replace remaining _ with . (for path separators)
    module_path = module_path.replace('_', '.')

    # Replace placeholder with ._ for private modules
    module_path = module_path.replace(':::PRIVATE:::', '._')

    # Extract function name and line number
    # Pattern: function_name_L<line_number>
    match = re.match(r'^(.+)_L(\d+)$', func_with_line)
    if match:
        func_name = match.group(1)
        line_number = int(match.group(2))
    else:
        func_name = func_with_line
        line_number = None

    return module_path, func_name, line_number


def parse_swebench_problem_name(problem_id):
    """
    Parse Swebench benchmark problem name like:
    'scikit-learn__scikit-learn-14544@@sklearn.compose._column_transformer.py@@fit_transform'
    'matplotlib__matplotlib-24403@@lib.matplotlib.axes._axes.py@@_parse_scatter_color_args'
    'sympy__sympy-13369@@sympy.polys.polyroots.py@@roots'

    Returns (module_path, function_name, line_number)
    """
    parts = problem_id.split('@@')
    if len(parts) != 3:
        return None, None, None

    repo_issue, file_path, func_name = parts

    # Remove .py extension and convert to module path
    module_path = file_path.replace('.py', '')

    # Handle special prefixes like 'lib.' for matplotlib, 'src.' for others
    if module_path.startswith('lib.'):
        module_path = module_path[4:]  # Remove 'lib.'
    elif module_path.startswith('src.'):
        module_path = module_path[4:]  # Remove 'src.'

    # Swebench doesn't have line numbers in the problem ID
    line_number = None

    return module_path, func_name, line_number

def generate_import_statement(module_path, func_name):
    """Generate import statement from module path and function name."""
    return f"from {module_path} import {func_name}"

def main():
    # Read the JSON file
    with open('/home/changshu/RE2-Bench/scripts/categorization/sampled_problems.json', 'r') as f:
        data = json.load(f)

    # Output directory
    output_dir = '/home/changshu/RE2-Bench/scripts/swebench_input_detection/tmp'
    os.makedirs(output_dir, exist_ok=True)

    # Process all categories
    for category, problems in data.items():
        for problem_id, info in problems.items():
            benchmark = info.get('benchmark')

            # Handle both Real and Swebench benchmarks
            # if benchmark == 'Real':
            #     module_path, func_name, line_number = parse_real_problem_name(problem_id)
            if benchmark == 'Swebench':
                module_path, func_name, line_number = parse_swebench_problem_name(problem_id)
            else:
                # Skip other benchmarks (cruxeval, Classeval, Avatar, HumanEval)
                continue

            if module_path is None:
                print(f"Skipping invalid problem: {problem_id}")
                continue

            import_statement = generate_import_statement(module_path, func_name)

            # Create the output file using the original problem_id as filename
            # Replace characters that are invalid in filenames
            safe_filename = problem_id.replace('/', '_').replace('\\', '_')
            output_file = os.path.join(output_dir, f"{safe_filename}.py")

            with open(output_file, 'w') as f:
                f.write(f"# Problem: {problem_id}\n")
                f.write(f"# Benchmark: {benchmark}\n")
                f.write(f"# Module: {module_path}\n")
                f.write(f"# Function: {func_name}\n")
                if line_number:
                    f.write(f"# Line: {line_number}\n")
                f.write(f"\n{import_statement}\n")

            print(f"Created: {output_file}")

if __name__ == '__main__':
    main()
