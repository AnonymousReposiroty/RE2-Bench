from utils import get_patch_commit
import subprocess
import os

def build_project(pid):
    commit_id, patch, repoName = get_patch_commit(pid)
    playground_path = "swebench-input-detection/repos"
    patch_path = "swebench-input-detection/repos/patch.diff"
    init_bash_path = "swebench-input-detection/src/create_projects/init_sklearn.sh"
    
    with open(patch_path, "w") as f:
        f.write(patch)

    print(playground_path)
    print(commit_id)
    print(repoName)
    subprocess.run(["bash", init_bash_path, playground_path, commit_id, repoName, "sklearn"])

def delete_project():
    playground_path = "swebench-input-detection/repos"
    print("Deleting sklearn project...")
    subprocess.run(["rm", "-rf", f"{playground_path}/sklearn"])
    subprocess.run(["rm", "-rf", ".venv-sk"])

if __name__ == "__main__":
    pid = "scikit-learn__scikit-learn-12682"
    # build_project(pid)
    delete_project()