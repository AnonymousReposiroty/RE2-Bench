#!/usr/bin/env python3
"""
Process JSON files from branch directories and create annotated code files.

For each JSON file:
1. Extract branch locations (line numbers) from branches.decisions
2. Read corresponding code file
3. Remove docstrings and comments
4. Add ## [BRANCH]taken=??[/BRANCH] annotations
5. Output to prompts/branch_prediction directory
"""

import json
import os
import re
from pathlib import Path


def extract_branch_lines(json_data):
    """Extract unique branch line numbers from JSON data."""
    branch_lines = set()
    if "branches" in json_data and "decisions" in json_data["branches"]:
        for decision in json_data["branches"]["decisions"]:
            if "line" in decision:
                branch_lines.add(decision["line"])
    return branch_lines


def remove_docstrings_and_comments(code_lines):
    """
    Remove docstrings and comments from code, returning processed lines
    with original line number mapping.
    """
    result = []
    line_mapping = {}  # Maps original line number to new line number

    in_docstring = False
    docstring_char = None
    i = 0
    new_line_num = 0

    while i < len(code_lines):
        line = code_lines[i]
        original_line_num = i + 1  # 1-indexed
        stripped = line.strip()

        # Check for docstring start
        if not in_docstring:
            # Check for triple quotes (docstring start)
            if '"""' in stripped or "'''" in stripped:
                # Determine which quote type
                if '"""' in stripped:
                    docstring_char = '"""'
                else:
                    docstring_char = "'''"

                # Check if it's a single-line docstring
                first_idx = stripped.find(docstring_char)
                remaining = stripped[first_idx + 3:]
                if docstring_char in remaining:
                    # Single-line docstring - skip this line entirely
                    i += 1
                    continue
                else:
                    # Multi-line docstring starts
                    in_docstring = True
                    i += 1
                    continue

            # Check for comment-only lines
            if stripped.startswith('#'):
                i += 1
                continue

            # Remove inline comments but keep the code
            # Be careful not to remove # inside strings
            processed_line = remove_inline_comment(line)

            if processed_line.strip():  # Only add non-empty lines
                new_line_num += 1
                result.append(processed_line)
                line_mapping[original_line_num] = new_line_num

        else:
            # We're inside a docstring, check for end
            if docstring_char in stripped:
                in_docstring = False
                docstring_char = None

        i += 1

    return result, line_mapping


def remove_inline_comment(line):
    """Remove inline comments from a line, preserving strings."""
    result = []
    in_string = False
    string_char = None
    i = 0

    while i < len(line):
        char = line[i]

        if not in_string:
            if char in ('"', "'"):
                # Check for triple quotes
                if line[i:i+3] in ('"""', "'''"):
                    result.append(line[i:i+3])
                    in_string = True
                    string_char = line[i:i+3]
                    i += 3
                    continue
                else:
                    in_string = True
                    string_char = char
            elif char == '#':
                # Found a comment - stop here
                break
        else:
            # In string, check for end
            if string_char in ('"""', "'''"):
                if line[i:i+3] == string_char:
                    result.append(line[i:i+3])
                    in_string = False
                    string_char = None
                    i += 3
                    continue
            elif char == string_char and (i == 0 or line[i-1] != '\\'):
                in_string = False
                string_char = None

        result.append(char)
        i += 1

    return ''.join(result).rstrip()


def process_code_with_branches(code_content, branch_lines):
    """
    Process code to remove docstrings/comments and add branch annotations.
    """
    lines = code_content.split('\n')

    # First, identify which original lines have branches
    # Then process the code
    result_lines = []

    in_docstring = False
    docstring_char = None

    for i, line in enumerate(lines):
        original_line_num = i + 1
        stripped = line.strip()

        # Check for docstring start/end
        if not in_docstring:
            if '"""' in stripped or "'''" in stripped:
                if '"""' in stripped:
                    docstring_char = '"""'
                else:
                    docstring_char = "'''"

                first_idx = stripped.find(docstring_char)
                remaining = stripped[first_idx + 3:]
                if docstring_char in remaining:
                    # Single-line docstring - skip
                    continue
                else:
                    in_docstring = True
                    continue

            # Skip comment-only lines
            if stripped.startswith('#'):
                continue

            # Preserve empty lines
            if not stripped:
                result_lines.append('')
                continue

            # Process the line
            processed_line = remove_inline_comment(line)

            if processed_line.strip():
                # Check if this line has a branch
                if original_line_num in branch_lines:
                    processed_line = processed_line.rstrip() + " ## [BRANCH]taken=??[/BRANCH]"
                result_lines.append(processed_line)
        else:
            if docstring_char in stripped:
                in_docstring = False
                docstring_char = None

    # Ensure trailing newline
    return '\n'.join(result_lines) + '\n'


def process_json_file(json_path, code_base_dir, output_dir):
    """Process a single JSON file and create the annotated output."""
    # Read JSON file
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # Extract branch lines
    branch_lines = extract_branch_lines(json_data)

    if not branch_lines:
        print(f"Warning: No branches found in {json_path}")
        return False

    # Construct code file path
    json_filename = os.path.basename(json_path)
    code_filename = json_filename.replace('.json', '.py')

    # Handle sample_XX -> cruxeval_XX naming
    if code_filename.startswith('sample_'):
        code_filename = code_filename.replace('sample_', 'cruxeval_', 1)
    if code_filename.startswith('ClassEval_') and '@' in code_filename:
        # Strip the ClassEval_XX@ prefix
        code_filename = code_filename.split('@', 1)[1]
    code_path = os.path.join(code_base_dir, code_filename)

    if not os.path.exists(code_path):
        print(f"Warning: Code file not found: {code_path}")
        return False

    # Read code file
    with open(code_path, 'r') as f:
        code_content = f.read()

    # Process code
    output_content = process_code_with_branches(code_content, branch_lines)

    # Write output
    output_filename = json_filename.replace('.json', '.txt')
    output_path = os.path.join(output_dir, output_filename)

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(output_content)

    print(f"Created: {output_path}")
    return True


def main():
    base_dir = "/home/changshu/RE2-Bench"
    code_base_dir = os.path.join(base_dir, "dataset/re2-bench/code")

    # Process difficult branch files
    difficult_input_dir = os.path.join(base_dir, "dataset/re2-bench/branch/difficult")
    difficult_output_dir = os.path.join(base_dir, "prompts/branch_prediction/difficult")

    # Process easy branch files
    easy_input_dir = os.path.join(base_dir, "dataset/re2-bench/branch/easy")
    easy_output_dir = os.path.join(base_dir, "prompts/branch_prediction-intermediate/easy")

    success_count = 0
    fail_count = 0

    for input_dir, output_dir in [
                                   (easy_input_dir, easy_output_dir)]:
        if not os.path.exists(input_dir):
            print(f"Warning: Input directory not found: {input_dir}")
            continue

        for filename in sorted(os.listdir(input_dir)):
            if filename.endswith('.json'):
                json_path = os.path.join(input_dir, filename)
                if process_json_file(json_path, code_base_dir, output_dir):
                    success_count += 1
                else:
                    fail_count += 1

    print(f"\nProcessed {success_count} files successfully, {fail_count} failures")


if __name__ == "__main__":
    main()
