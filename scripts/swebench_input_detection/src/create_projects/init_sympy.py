from .utils import get_patch_commit
import subprocess
import os

def build_project(pid):
    commit_id, patch, repoName = get_patch_commit(pid)
    playground_path = "swebench-input-detection/repos"
    patch_path = "swebench-input-detection/repos/patch.diff"
    init_bash_path = "swebench-input-detection/src/create_projects/init_sympy.sh"
    
    with open(patch_path, "w") as f:
        f.write(patch)

    subprocess.run(["bash", init_bash_path, playground_path, commit_id, repoName, "sympy"])

def delete_project():
    playground_path = "swebench-input-detection/repos"
    print("Deleting sympy project...")
    subprocess.run(["rm", "-rf", f"{playground_path}/sympy"])

if __name__ == "__main__":
    pid = "sympy__sympy-15222"
    build_project(pid)
    delete_project()