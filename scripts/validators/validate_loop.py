import re
import ast


def parse_loop(problem_id, difficulty, model_id):
    """Parse the loop information from the problem ID.

    Args:
        problem_id (str): The problem identifier string.
        difficulty (str): The difficulty level (e.g., "easy", "difficult").
        model_id (str): The model identifier.
    Returns:
        dict: A dictionary containing the variable states, keyed by line number.
    """
    prediction_path = f"../results/loop_prediction/{model_id}/{difficulty}/{problem_id}.txt"

    # Read the file content
    with open(prediction_path, 'r') as f:
        content = f.read()

    # Extract content between [ANSWER] and [/ANSWER]
    answer_match = re.search(r'\[ANSWER\](.*?)\[/ANSWER\]', content, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1)
    else:
        # Fallback: match from [ANSWER] to end of file if [/ANSWER] is missing
        answer_match = re.search(r'\[ANSWER\](.*)', content, re.DOTALL)
        if not answer_match:
            return {}
        answer_content = answer_match.group(1)

    answer_content = answer_match.group(1).strip()

    # Parse each line for variable states
    result = {}
    state_pattern = re.compile(r'\[STATE\](.+?)=(.+?)\[/STATE\]')

    for line in answer_content.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Extract line number (first token before space)
        line_parts = line.split(' ', 1)
        if not line_parts:
            continue

        line_num = line_parts[0]

        # Find all [STATE]var=value[/STATE] patterns in this line
        states = state_pattern.findall(line)
        if states:
            line_states = {}
            for var_name, var_value in states:
                # Try to parse the value as a Python literal
                try:
                    parsed_value = ast.literal_eval(var_value)
                except (ValueError, SyntaxError):
                    # If parsing fails, keep as string
                    parsed_value = var_value
                line_states[var_name] = parsed_value

            if line_states:
                result[line_num] = line_states

    return result


def read_ground_truth(problem_id, difficulty):
    """Read the ground truth loop information from a file.

    Args:
        problem_id (str): The problem identifier string.
        difficulty (str): The difficulty level (e.g., "easy", "difficult").
    Returns:
        dict: A dictionary containing the ground truth variable states, keyed by line number.
    """
    import json

    ground_truth_path = f"../dataset/re2-bench/loop/{difficulty}/{problem_id}.json"

    with open(ground_truth_path, 'r') as f:
        data = json.load(f)

    result = {}

    # Process each loop in the "loops" dictionary
    for loop_id, loop_data in data.get("loops", {}).items():
        line_num = str(loop_data.get("line"))
        iterations = loop_data.get("iterations", [])

        if not iterations:
            continue

        line_states = {}

        # Aggregate loop variable values across all iterations
        for iteration in iterations:
            loop_vars = iteration.get("loop_variables", {})
            for var_name, var_value in loop_vars.items():
                if var_name not in line_states:
                    line_states[var_name] = []
                line_states[var_name].append(var_value)

        # Aggregate control variable values across all iterations
        ctrl_values_by_name = {}
        for iteration in iterations:
            control_var = iteration.get("control_variable", {})
            ctrl_name = control_var.get("name")
            ctrl_value = control_var.get("value")
            if ctrl_name:
                if ctrl_name not in ctrl_values_by_name:
                    ctrl_values_by_name[ctrl_name] = []
                ctrl_values_by_name[ctrl_name].append(ctrl_value)

        # Only include control variables that have at least one non-null and non-empty value
        def is_meaningful(v):
            """Check if a value is meaningful (not null and not empty)."""
            if v is None:
                return False
            # Check for empty collections (list, dict, str, tuple, set)
            if isinstance(v, (list, dict, str, tuple, set)) and len(v) == 0:
                return False
            return True

        for ctrl_name, ctrl_values in ctrl_values_by_name.items():
            if any(is_meaningful(v) for v in ctrl_values):
                line_states[ctrl_name] = ctrl_values

        if line_states:
            result[line_num] = line_states

    return result


def normalize_value(value):
    """Recursively convert tuples to lists for consistent comparison.

    Args:
        value: Any value that may contain nested tuples/lists.
    Returns:
        The value with all tuples converted to lists.
    """
    if isinstance(value, tuple):
        return [normalize_value(item) for item in value]
    elif isinstance(value, list):
        return [normalize_value(item) for item in value]
    elif isinstance(value, dict):
        return {k: normalize_value(v) for k, v in value.items()}
    else:
        return value


def compare_loop(prediction, ground_truth):
    """Compare the prediction with the ground truth.

    Args:
        prediction (dict): The predicted variable states from parse_loop.
        ground_truth (dict): The ground truth variable states from read_ground_truth.
    Returns:
        tuple: A tuple containing:
            - overall_score (int): 0 or 1, 1 if all variables in all loops are correct
            - partial_score (float): Fraction of loops where all variables are correctly predicted
    """
    # Get all line numbers (loops) from ground truth
    all_lines = list(ground_truth.keys())
    total_loops = len(all_lines)

    if total_loops == 0:
        return 1, 1.0

    correct_loops = 0

    for line_num in all_lines:
        gt_vars = ground_truth.get(line_num, {})
        pred_vars = prediction.get(line_num, {})

        # Check if this loop is completely correct
        loop_correct = True

        # If the line is missing in prediction, the loop is incorrect
        if line_num not in prediction:
            loop_correct = False
        else:
            # Check each variable in ground truth
            for var_name, gt_value in gt_vars.items():
                if var_name not in pred_vars:
                    loop_correct = False
                    break
                # Normalize both values before comparison (tuples -> lists)
                normalized_gt = normalize_value(gt_value)
                normalized_pred = normalize_value(pred_vars[var_name])
                if normalized_pred != normalized_gt:
                    loop_correct = False
                    break

        if loop_correct:
            correct_loops += 1

    # overall_score: 1 if all loops are correct, 0 otherwise
    overall_score = 1 if correct_loops == total_loops else 0

    # partial_score: fraction of loops that are completely correct
    partial_score = correct_loops / total_loops

    return overall_score, partial_score


def validate_loop(problem_id, difficulty, model_id):
    """Validate the loop prediction against ground truth.

    Args:
        problem_id (str): The problem identifier string.
        difficulty (str): The difficulty level (e.g., "easy", "difficult").
        model_id (str): The model identifier.
    Returns:
        tuple: A tuple containing:
            - overall_score (int): 0 or 1, 1 if all variables in all loops are correct
            - partial_score (float): Fraction of loops where all variables are correctly predicted
    """
    prediction = parse_loop(problem_id, difficulty, model_id)
    ground_truth = read_ground_truth(problem_id, difficulty)
    return compare_loop(prediction, ground_truth)


def evaluate_model(model_id, difficulties=None):
    """Evaluate a model on all problems across specified difficulty levels.

    Args:
        model_id (str): The model identifier.
        difficulties (list): List of difficulty levels to evaluate. Defaults to ["easy", "difficult"].
    Returns:
        dict: A dictionary containing evaluation results per difficulty and overall.
    """
    import os
    from glob import glob

    if difficulties is None:
        difficulties = ["easy", "difficult"]

    results = {
        "model_id": model_id,
        "by_difficulty": {},
        "overall": {
            "total_problems": 0,
            "successful_problems": 0,
            "failed_problems": 0,
            "overall_score_sum": 0,
            "partial_score_sum": 0.0,
            "overall_accuracy": 0.0,
            "partial_accuracy": 0.0,
        }
    }

    for difficulty in difficulties:
        # Find all ground truth files for this difficulty
        gt_dir = f"../dataset/re2-bench/loop/{difficulty}"
        pred_dir = f"../results/loop_prediction/{model_id}/{difficulty}"

        if not os.path.exists(gt_dir):
            print(f"Warning: Ground truth directory not found: {gt_dir}")
            continue

        if not os.path.exists(pred_dir):
            print(f"Warning: Prediction directory not found: {pred_dir}")
            continue

        gt_files = glob(os.path.join(gt_dir, "*.json"))
        problem_ids = [os.path.basename(f).replace(".json", "") for f in gt_files]

        difficulty_results = {
            "total_problems": 0,
            "successful_problems": 0,
            "failed_problems": 0,
            "overall_score_sum": 0,
            "partial_score_sum": 0.0,
            "problems": []
        }

        for problem_id in problem_ids:
            pred_file = os.path.join(pred_dir, f"{problem_id}.txt")
            if not os.path.exists(pred_file):
                difficulty_results["failed_problems"] += 1
                difficulty_results["total_problems"] += 1
                continue

            try:
                overall_score, partial_score = validate_loop(problem_id, difficulty, model_id)
                difficulty_results["problems"].append({
                    "problem_id": problem_id,
                    "overall_score": overall_score,
                    "partial_score": partial_score,
                    "status": "success"
                })
                difficulty_results["overall_score_sum"] += overall_score
                difficulty_results["partial_score_sum"] += partial_score
                difficulty_results["successful_problems"] += 1
            except Exception as e:
                difficulty_results["problems"].append({
                    "problem_id": problem_id,
                    "status": "error",
                    "error": str(e)
                })
                difficulty_results["failed_problems"] += 1

            difficulty_results["total_problems"] += 1

        # Calculate accuracy for this difficulty
        if difficulty_results["successful_problems"] > 0:
            difficulty_results["overall_accuracy"] = (
                difficulty_results["overall_score_sum"] / difficulty_results["successful_problems"]
            )
            difficulty_results["partial_accuracy"] = (
                difficulty_results["partial_score_sum"] / difficulty_results["successful_problems"]
            )
        else:
            difficulty_results["overall_accuracy"] = 0.0
            difficulty_results["partial_accuracy"] = 0.0

        results["by_difficulty"][difficulty] = difficulty_results

        # Update overall results
        results["overall"]["total_problems"] += difficulty_results["total_problems"]
        results["overall"]["successful_problems"] += difficulty_results["successful_problems"]
        results["overall"]["failed_problems"] += difficulty_results["failed_problems"]
        results["overall"]["overall_score_sum"] += difficulty_results["overall_score_sum"]
        results["overall"]["partial_score_sum"] += difficulty_results["partial_score_sum"]

    # Calculate overall accuracy
    if results["overall"]["successful_problems"] > 0:
        results["overall"]["overall_accuracy"] = (
            results["overall"]["overall_score_sum"] / results["overall"]["successful_problems"]
        )
        results["overall"]["partial_accuracy"] = (
            results["overall"]["partial_score_sum"] / results["overall"]["successful_problems"]
        )

    return results


def print_evaluation_results(results):
    """Print evaluation results in a formatted way.

    Args:
        results (dict): The evaluation results from evaluate_model.
    """
    print(f"\n{'='*60}")
    print(f"Model: {results['model_id']}")
    print(f"{'='*60}")

    for difficulty, diff_results in results["by_difficulty"].items():
        print(f"\n--- {difficulty.upper()} ---")
        print(f"  Total Problems:      {diff_results['total_problems']}")
        print(f"  Successful:          {diff_results['successful_problems']}")
        print(f"  Failed/Missing:      {diff_results['failed_problems']}")
        print(f"  Overall Accuracy:    {diff_results['overall_accuracy']:.2%}")
        print(f"  Partial Accuracy:    {diff_results['partial_accuracy']:.2%}")

    print(f"\n--- OVERALL ---")
    print(f"  Total Problems:      {results['overall']['total_problems']}")
    print(f"  Successful:          {results['overall']['successful_problems']}")
    print(f"  Failed/Missing:      {results['overall']['failed_problems']}")
    print(f"  Overall Accuracy:    {results['overall']['overall_accuracy']:.2%}")
    print(f"  Partial Accuracy:    {results['overall']['partial_accuracy']:.2%}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate loop predictions against ground truth.")
    parser.add_argument("--model", "-m", type=str, default="gemini-3-pro-preview-reasoning",
                        help="Model ID to evaluate")
    parser.add_argument("--difficulty", "-d", type=str, choices=["easy", "difficult", "all"],
                        default="all", help="Difficulty level to evaluate")
    parser.add_argument("--problem", "-p", type=str, default=None,
                        help="Specific problem ID to evaluate (requires --difficulty)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed results for each problem")

    args = parser.parse_args()

    if args.problem:
        # Evaluate a single problem
        if args.difficulty == "all":
            print("Error: --difficulty must be 'easy' or 'difficult' when evaluating a single problem")
            exit(1)

        print(f"Evaluating problem: {args.problem} ({args.difficulty})")
        try:
            overall_score, partial_score = validate_loop(args.problem, args.difficulty, args.model)
            print(f"  Overall Score: {overall_score}")
            print(f"  Partial Score: {partial_score:.2f}")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        # Evaluate all problems
        difficulties = ["easy", "difficult"] if args.difficulty == "all" else [args.difficulty]
        results = evaluate_model(args.model, difficulties)
        print_evaluation_results(results)

        if args.verbose:
            print("\n--- Detailed Results ---")
            for difficulty, diff_results in results["by_difficulty"].items():
                print(f"\n{difficulty.upper()}:")
                for prob in diff_results["problems"]:
                    if prob["status"] == "success":
                        print(f"  {prob['problem_id']}: overall={prob['overall_score']}, partial={prob['partial_score']:.2f}")
                    else:
                        print(f"  {prob['problem_id']}: ERROR - {prob.get('error', 'unknown')}")