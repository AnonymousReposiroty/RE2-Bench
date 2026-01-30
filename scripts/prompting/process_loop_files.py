#!/usr/bin/env python3
"""
Process loop JSON files and generate annotated code prompts.
"""
import json
import os
import re
import ast
import io
import tokenize


def remove_comments_and_docstrings(source_code):
    """
    Remove docstrings and comments from Python source code.
    Uses tokenize-based approach for more accurate results.
    """
    try:
        # Use tokenize to remove comments and string literals that are docstrings
        tokens = list(tokenize.generate_tokens(io.StringIO(source_code).readline))

        # First pass: identify docstring positions
        docstring_positions = set()

        for i, tok in enumerate(tokens):
            toktype = tok.type

            if toktype == tokenize.STRING:
                # Check if this is a docstring (standalone string at start of body)
                # Look backwards to see if the previous non-NL/INDENT/NEWLINE token
                # is a colon (start of body) or nothing (module level)
                prev_meaningful = None
                for j in range(i - 1, -1, -1):
                    ptok = tokens[j]
                    if ptok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                                          tokenize.DEDENT, tokenize.COMMENT, tokenize.ENCODING):
                        prev_meaningful = ptok
                        break

                # Check if this is a docstring
                if prev_meaningful is None:
                    # Module-level docstring
                    for row in range(tok.start[0], tok.end[0] + 1):
                        docstring_positions.add(row)
                elif prev_meaningful.type == tokenize.OP and prev_meaningful.string == ':':
                    # Function/class docstring
                    for row in range(tok.start[0], tok.end[0] + 1):
                        docstring_positions.add(row)

        # Second pass: rebuild code without docstrings and comments
        lines = source_code.split('\n')
        result_lines = []

        for line_no, line in enumerate(lines, 1):
            if line_no in docstring_positions:
                continue

            # Remove comments from the line
            new_line = line
            in_string = False
            string_char = None
            escape_next = False

            for j, char in enumerate(line):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char in ('"', "'"):
                    if not in_string:
                        # Check for triple quotes
                        if line[j:j+3] in ('"""', "'''"):
                            in_string = True
                            string_char = line[j:j+3]
                        else:
                            in_string = True
                            string_char = char
                    elif len(string_char) == 3 and line[j:j+3] == string_char:
                        in_string = False
                        string_char = None
                    elif len(string_char) == 1 and char == string_char:
                        in_string = False
                        string_char = None
                elif char == '#' and not in_string:
                    new_line = line[:j].rstrip()
                    break

            # Only add non-empty lines or preserve structure
            if new_line.strip():
                result_lines.append(new_line)
            elif result_lines and result_lines[-1].strip():
                # Keep single blank line for readability
                result_lines.append('')

        # Remove trailing blank lines
        while result_lines and not result_lines[-1].strip():
            result_lines.pop()

        # Remove leading blank lines
        while result_lines and not result_lines[0].strip():
            result_lines.pop(0)

        return '\n'.join(result_lines)

    except tokenize.TokenizeError:
        # Fallback: just return the original code
        return source_code


def extract_loops_from_json(json_data):
    """
    Extract loop information from JSON data.
    Returns a list of dicts with loop_vars (list), control_var, line info.
    Excludes loops where state/value is null.
    Handles both singular (control_var/control_variable) and plural (control_vars/control_variables) naming.
    Reads individual loop variable names from iterations.loop_variables keys.
    """
    loops_info = []
    loops = json_data.get('loops', {})

    for loop_id, loop_data in loops.items():
        loop_var = loop_data.get('loop_var')
        # Handle both control_var (singular) and control_vars (plural list)
        control_var = loop_data.get('control_var')
        control_vars = loop_data.get('control_vars', [])
        # If control_var is not set but control_vars is a list, use the first item
        if not control_var and control_vars:
            control_var = control_vars[0] if isinstance(control_vars, list) else control_vars
        line = loop_data.get('line', -1)
        code = loop_data.get('code', '')

        # Check if there are valid iterations with non-null values
        iterations = loop_data.get('iterations', [])
        has_valid_iteration = False
        loop_var_names = []  # Individual loop variable names from iterations

        for iteration in iterations:
            loop_variables = iteration.get('loop_variables', {})
            # Handle both control_variable (singular dict) and control_variables (plural dict)
            control_variable = iteration.get('control_variable', {})
            control_variables = iteration.get('control_variables', {})

            # Check loop variables are not null
            loop_vars_valid = any(v is not None for v in loop_variables.values()) if loop_variables else False

            # Extract individual loop variable names from the first valid iteration
            if loop_variables and not loop_var_names:
                loop_var_names = list(loop_variables.keys())

            # Check control variable is not null (handle both singular and plural)
            control_valid = False
            if control_variable:
                control_value = control_variable.get('value') if isinstance(control_variable, dict) else control_variable
                control_valid = control_value is not None
            if not control_valid and control_variables:
                # control_variables is a dict like {"j": 0}
                control_valid = any(v is not None for v in control_variables.values())

            if loop_vars_valid or control_valid:
                has_valid_iteration = True
                break

        # Include loop if it has valid iteration AND has either loop_var or control_var
        if has_valid_iteration and (loop_var or control_var):
            loops_info.append({
                'loop_var': loop_var,
                'loop_vars': loop_var_names,  # Individual variable names from iterations
                'control_var': control_var,
                'line': line,
                'code': code
            })

    return loops_info


def find_loop_line_in_code(code_lines, loop_code, original_line):
    """
    Find the line number in cleaned code that matches the loop.
    """
    # Normalize the loop code for matching
    loop_code_normalized = loop_code.strip().rstrip(':')

    for i, line in enumerate(code_lines):
        line_stripped = line.strip().rstrip(':')
        # Check if this line contains the loop pattern
        if 'for ' in line or 'while ' in line:
            # Try to match the loop variable pattern
            if loop_code_normalized in line_stripped or line_stripped in loop_code_normalized:
                return i
            # Also try matching just the for/while part
            if loop_code.split()[0] in line and loop_code.split()[1] in line:
                return i

    return -1


def is_comprehension_line(line):
    """
    Check if a line contains a comprehension (list, dict, set, or generator).
    """
    # Check for comprehension brackets with 'for' inside
    # List comprehension: [... for ... in ...]
    # Dict/Set comprehension: {... for ... in ...}
    # Generator expression: (... for ... in ...)
    stripped = line.strip()

    # Look for patterns like [... for, {... for, (... for
    # Make sure 'for' appears after an opening bracket
    for open_bracket, close_bracket in [('[', ']'), ('{', '}'), ('(', ')')]:
        if open_bracket in stripped and ' for ' in stripped:
            # Find the position of the bracket and 'for'
            bracket_pos = stripped.find(open_bracket)
            for_pos = stripped.find(' for ')
            if bracket_pos < for_pos:
                return True
    return False


def annotate_code_with_loops(cleaned_code, loops_info):
    """
    Insert state annotations right after each loop line.
    Format: ## [STATE]var1=??[/STATE] [STATE]var2=??[/STATE][STATE]control_var=??[/STATE]
    Uses individual loop variable names from iterations when available.
    Handles both regular for/while loops and comprehensions (list, dict, set, generator).
    """
    lines = cleaned_code.split('\n')

    # For each loop, find its position and insert annotation
    # We need to track insertions to adjust line numbers
    insertions = []  # List of (line_index, annotation)

    for loop in loops_info:
        loop_var = loop['loop_var']
        loop_vars = loop.get('loop_vars', [])  # Individual variable names from iterations
        control_var = loop['control_var']
        loop_code = loop['code']

        # Create the annotation with available variables
        # Prefer individual loop_vars from iterations over the combined loop_var string
        state_parts = []
        if loop_vars:
            # Use individual variable names from iterations
            for var_name in loop_vars:
                state_parts.append(f"[STATE]{var_name}=??[/STATE]")
        elif loop_var:
            # Fallback to the original loop_var string
            state_parts.append(f"[STATE]{loop_var}=??[/STATE]")
        if control_var:
            state_parts.append(f"[STATE]{control_var}=??[/STATE]")
        annotation = "## " + " ".join(state_parts)

        # Find the loop line in the cleaned code
        matched = False
        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Match regular for/while loops (line starts with for/while)
            if line_stripped.startswith('for ') or line_stripped.startswith('while '):
                # Check if this is the right loop by matching the loop variable
                # For "for x in ..." pattern
                if loop_var and (f'for {loop_var} ' in line_stripped or f'for {loop_var},' in line_stripped):
                    insertions.append((i, annotation, line))
                    matched = True
                    break
                # For while loops, check if the loop code matches
                if loop_code and loop_code.strip().rstrip(':') in line_stripped:
                    insertions.append((i, annotation, line))
                    matched = True
                    break

            # Match comprehensions (list, dict, set, generator expressions)
            # These have 'for' in the middle of the line, not at the start
            elif is_comprehension_line(line_stripped):
                # Check if this comprehension matches our loop variable
                # Pattern: "for {loop_var} in" or "for {loop_var},"
                if loop_var and (f' for {loop_var} in ' in line_stripped or
                                 f' for {loop_var},' in line_stripped):
                    insertions.append((i, annotation, line))
                    matched = True
                    break
                # Also try matching the loop code directly
                if loop_code:
                    # Normalize both for comparison
                    loop_code_normalized = loop_code.strip()
                    if loop_code_normalized in line_stripped or line_stripped in loop_code_normalized:
                        insertions.append((i, annotation, line))
                        matched = True
                        break

    # Apply insertions (add annotation right after the loop line, on the same line)
    result_lines = []
    loop_lines_annotated = set()

    for insertion in insertions:
        loop_lines_annotated.add(insertion[0])

    for i, line in enumerate(lines):
        if i in loop_lines_annotated:
            # Find the annotation for this line
            for ins in insertions:
                if ins[0] == i:
                    # Add annotation on the same line after the code
                    result_lines.append(f"{line}{ins[1]}")
                    break
        else:
            result_lines.append(line)

    return '\n'.join(result_lines)


def get_code_file_path(json_filename, code_dir):
    """
    Get the corresponding code file path from JSON filename.
    Handles different naming conventions:
    - ClassEval_XX@ClassName.method.json -> ClassName.method.py
    - sample_XXX.json -> cruxeval_XXX.py
    - Regular files: filename.json -> filename.py
    """
    # Remove .json extension
    base_name = json_filename.replace('.json', '')

    # Handle ClassEval naming: ClassEval_XX@ClassName.method -> ClassName.method
    if base_name.startswith('ClassEval_') and '@' in base_name:
        # Strip the ClassEval_XX@ prefix
        base_name = base_name.split('@', 1)[1]

    # Handle sample -> cruxeval naming
    if base_name.startswith('sample_'):
        base_name = base_name.replace('sample_', 'cruxeval_', 1)

    # Add .py extension
    code_filename = base_name + '.py'
    return os.path.join(code_dir, code_filename)


def process_file(json_path, code_dir, output_dir):
    """
    Process a single JSON file and create the output prompt file.
    """
    # Read JSON file
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # Extract loop information
    loops_info = extract_loops_from_json(json_data)

    if not loops_info:
        print(f"  Skipping {json_path}: No valid loop variables found")
        return False

    # Get code file path
    json_filename = os.path.basename(json_path)
    code_path = get_code_file_path(json_filename, code_dir)

    if not os.path.exists(code_path):
        print(f"  Skipping {json_path}: Code file not found at {code_path}")
        return False

    # Read code file
    with open(code_path, 'r') as f:
        code = f.read()

    # Remove comments and docstrings
    cleaned_code = remove_comments_and_docstrings(code)

    # Annotate the code with loop state markers
    annotated_code = annotate_code_with_loops(cleaned_code, loops_info)

    # Write output file
    output_filename = json_filename.replace('.json', '.txt')
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w') as f:
        f.write(annotated_code + '\n')

    return True


def main():
    base_dir = '/home/changshu/RE2-Bench/dataset/re2-bench'
    code_dir = os.path.join(base_dir, 'code')
    prompts_base = '/home/changshu/RE2-Bench/prompts/loop_prediction'

    # Ensure output directories exist
    os.makedirs(os.path.join(prompts_base, 'difficult'), exist_ok=True)
    os.makedirs(os.path.join(prompts_base, 'easy'), exist_ok=True)

    # Process difficult files
    difficult_input_dir = os.path.join(base_dir, 'loop', 'difficult')
    difficult_output_dir = os.path.join(prompts_base, 'difficult')

    print("Processing difficult files...")
    processed = 0
    for filename in sorted(os.listdir(difficult_input_dir)):
        if filename.endswith('.json'):
            json_path = os.path.join(difficult_input_dir, filename)
            if process_file(json_path, code_dir, difficult_output_dir):
                processed += 1
    print(f"  Processed {processed} difficult files")

    # Process easy files
    easy_input_dir = os.path.join(base_dir, 'loop', 'easy')
    easy_output_dir = os.path.join(prompts_base, 'easy')

    print("Processing easy files...")
    processed = 0
    for filename in sorted(os.listdir(easy_input_dir)):
        if filename.endswith('.json'):
            json_path = os.path.join(easy_input_dir, filename)
            if process_file(json_path, code_dir, easy_output_dir):
                processed += 1
    print(f"  Processed {processed} easy files")


if __name__ == '__main__':
    main()
