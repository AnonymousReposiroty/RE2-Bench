import csv
import subprocess
import sys

def split_results_by_rs(model_id, proje_prefix):
    rs1_list = []
    rs0_list = []
    csv_path = f"../results/validations/{model_id}_input.csv"
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_name = row.get("case_name", "")
            if not case_name.startswith(proje_prefix):
                continue

            rs_val = row.get("rs", "0")
            try:
                rs_int = int(float(rs_val))
            except ValueError:
                # skip rows with invalid rs
                continue

            if rs_int == 1:
                rs1_list.append(case_name[:-6])
            elif rs_int == 0:
                rs0_list.append(case_name[:-6])
    return rs1_list, rs0_list


def recover_name(problem_id):
    m_name = problem_id.split(".")[-1]
    base_name = ".".join(problem_id.split(".")[:-1])
    return f"{base_name}.py@@{m_name}"

if __name__ == "__main__":
    model = sys.argv[1]
    prefix= sys.argv[2]
    rs1_list, rs0_list = split_results_by_rs(model, prefix)
    count = 0
    # rs0_list = ["scikit-learn__scikit-learn-15625@@sklearn.metrics._classification.confusion_matrix", "scikit-learn__scikit-learn-13554@@sklearn.metrics.pairwise.euclidean_distances"]
    for i in rs0_list:
        i = recover_name(i)
        print(i)
        
        result = subprocess.run(
            ["python", "-m", "swebench_input_detection.src.run_test", model, i],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, 
            text=True                 
        )
        print("STDOUT:", result.stdout)
        if "Test failed for" in result.stdout:
            print(f"Failed test case with rs=0: {i}")
        else:
            count += 1
            print(f"Passed test case with rs=0: {i}")
    print(f"Total passed test cases with rs=0: {count} out of {len(rs0_list)}")
    print(f"Total passed test cases with rs=1: {len(rs1_list)}")