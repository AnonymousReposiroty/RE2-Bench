import re
import ast


def parse_branch(problem_id, difficulty, model_id):
    """Parse the branch information from the problem ID.

    Args:
        problem_id (str): The problem identifier string.
        difficulty (str): The difficulty level (e.g., "easy", "difficult").
        model_id (str): The model identifier.
    Returns:
        dict: A dictionary containing the variable states, keyed by line number.
    """
    prediction_path = f"../results/branch_prediction/{model_id}/{difficulty}/{problem_id}.txt"
    # Read the file content
    with open(prediction_path, 'r') as f:
        content = f.read()

    # Extract content between [ANSWER] and [/ANSWER]
    # First try with closing tag, then fallback to [ANSWER] to end of file
    answer_match = re.search(r'\[ANSWER\](.*?)\[/ANSWER\]', content, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1)
    else:
        # Fallback: match from [ANSWER] to end of file if [/ANSWER] is missing
        answer_match = re.search(r'\[ANSWER\](.*)', content, re.DOTALL)
        if not answer_match:
            return {}
        answer_content = answer_match.group(1)
    result = {}

    # Find all lines with branch annotations
    # Pattern matches: line_number ... ## [BRANCH]taken=[...][/BRANCH]
    branch_pattern = r'^(\d+)\s+.*##\s*\[BRANCH\]taken=\[(.*?)\]\[/BRANCH\]'

    for line in answer_content.split('\n'):
        match = re.search(branch_pattern, line)
        if match:
            line_number = match.group(1)
            taken_values = match.group(2)

            # Parse the taken values (can be Y, N, 'Y', 'N', or comma-separated list)
            bool_list = []
            # Split by comma, handling potential spaces
            values = [v.strip().strip("'\"") for v in taken_values.split(',')]
            for val in values:
                if val.upper() == 'Y':
                    bool_list.append(True)
                elif val.upper() == 'N':
                    bool_list.append(False)

            if bool_list:
                result[line_number] = bool_list

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

    ground_truth_path = f"../dataset/re2-bench/branch/{difficulty}/{problem_id}.json"

    with open(ground_truth_path, 'r') as f:
        data = json.load(f)

    result = {}
    decisions = data.get("branches", {}).get("decisions", [])

    for decision in decisions:
        line_number = str(decision["line"])
        taken = decision["taken"]

        if line_number not in result:
            result[line_number] = []
        result[line_number].append(taken)

    return result

def compare_branch(prediction, ground_truth):
    """Compare the prediction with the ground truth.

    Args:
        prediction (dict): The predicted branch states from parse_branch.
            Format: {line_number: [bool, bool, ...]}
        ground_truth (dict): The ground truth branch states from read_ground_truth.
            Format: {line_number: [bool, bool, ...]}
    Returns:
        tuple: A tuple containing:
            - overall_score (int): 0 or 1, 1 if all branches match the ground truth
            - partial_score (float): Fraction of branches where the prediction matches
    """
    # Get all line numbers (branches) from ground truth
    all_lines = list(ground_truth.keys())
    total_branches = len(all_lines)

    if total_branches == 0:
        return 1, 1.0

    correct_branches = 0

    for line_num in all_lines:
        gt_values = ground_truth.get(line_num, [])
        pred_values = prediction.get(line_num, [])

        # Check if this branch is correctly predicted
        if pred_values == gt_values:
            correct_branches += 1

    partial_score = correct_branches / total_branches
    overall_score = 1 if correct_branches == total_branches else 0

    return overall_score, partial_score


def validate_branch(problem_id, difficulty, model_id):
    """Validate the branch prediction against ground truth.

    Args:
        problem_id (str): The problem identifier string.
        difficulty (str): The difficulty level (e.g., "easy", "difficult").
        model_id (str): The model identifier.
    Returns:
        tuple: A tuple containing:
            - overall_score (int): 0 or 1, 1 if all branches match the ground truth
            - partial_score (float): Fraction of branches where the prediction matches
    """
    prediction = parse_branch(problem_id, difficulty, model_id)
    ground_truth = read_ground_truth(problem_id, difficulty)
    return compare_branch(prediction, ground_truth)


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
        gt_dir = f"../dataset/re2-bench/branch/{difficulty}"
        pred_dir = f"../results/branch_prediction/{model_id}/{difficulty}"

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
                overall_score, partial_score = validate_branch(problem_id, difficulty, model_id)
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

    parser = argparse.ArgumentParser(description="Validate branch predictions against ground truth.")
    parser.add_argument("--model", "-m", type=str, default="claude-haiku-4.5",
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
            overall_score, partial_score = validate_branch(args.problem, args.difficulty, args.model)
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