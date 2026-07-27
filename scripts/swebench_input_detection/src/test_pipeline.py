from swebench_input_detection.src.create_projects import utils
import subprocess
import os
import sys


def runtime_validation(model_id,problem_id):
    commit_id, patch, repoName = utils.get_patch_commit(problem_id.split("@@")[0])
    playground_path = "swebench_input_detection/repos"
    if not os.path.exists(playground_path):
        os.makedirs(playground_path)
    patch_path = "swebench_input_detection/patch.diff"
    
    ## Clone the repo, apply the patch, create venv and install pkgs
    if problem_id.startswith("sympy"):
        init_bash_path = "swebench_input_detection/src/create_projects/init_sympy.sh"
        repo = "sympy"
    elif problem_id.startswith("scikit-learn"):
        init_bash_path = "swebench_input_detection/src/create_projects/init_sklearn.sh"
        repo = "sklearn"
    elif problem_id.startswith("sphinx-doc"):
        init_bash_path = "swebench_input_detection/src/create_projects/init_sphinx.sh"
        repo = "sphinx"
    elif problem_id.startswith("django__django"):
        init_bash_path = "swebench_input_detection/src/create_projects/init_django.sh"
        repo = "django"
    elif problem_id.startswith("matplotlib"):
        init_bash_path = "swebench_input_detection/src/create_projects/init_matplotlib.sh"
        repo = "matplotlib"
    elif problem_id.startswith("astropy__astropy"):
        init_bash_path = "swebench_input_detection/src/create_projects/init_astropy.sh"
        repo = "astropy"
    elif problem_id.startswith("mwaskom__seaborn"):
        init_bash_path = "swebench_input_detection/src/create_projects/init_seaborn.sh"
        repo = "seaborn"
    elif problem_id.startswith("psf__requests"):
        init_bash_path = "swebench_input_detection/src/create_projects/init_requests.sh"
        repo = "requests"
    elif problem_id.startswith("pydata__xarray"):
        init_bash_path = "swebench_input_detection/src/create_projects/init_xarray.sh"
        repo = "xarray"
    
    with open(patch_path, "w") as f:
        f.write(patch)

    subprocess.run(["bash", init_bash_path, playground_path, commit_id, repoName, repo])
    
    
    ## run tests
    print("run test...")
    
    result = subprocess.run(
            [".venv/bin/python", "-m", "swebench_input_detection.src.run_test", model_id, problem_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, 
            text=True                 
        )

    
    if "Test failed for" in result.stdout:
        print(f"Failed test case: {problem_id}")
    else:
        print(f"Passed test case: {problem_id}")
    
    ## clean up
    print("delete repo and venv")
    subprocess.run(["rm", "-rf", "swebench_input_detection/repos"])
    subprocess.run(["rm", "-rf", ".venv"])

if __name__ == "__main__":
    model = sys.argv[1]
    prob= sys.argv[2]
    runtime_validation(model, prob)