import subprocess
from datasets import load_dataset

repo_to_top_folder = {
    "django/django": "django",
    "sphinx-doc/sphinx": "sphinx",
    "scikit-learn/scikit-learn": "scikit-learn",
    "sympy/sympy": "sympy",
    "pytest-dev/pytest": "pytest",
    "matplotlib/matplotlib": "matplotlib",
    "astropy/astropy": "astropy",
    "pydata/xarray": "xarray",
    "mwaskom/seaborn": "seaborn",
    "psf/requests": "requests",
    "pylint-dev/pylint": "pylint",
    "pallets/flask": "flask",
}


def clone_repo(repo_name, repo_playground):
    try:

        print(
            f"Cloning repository from https://github.com/{repo_name}.git to {repo_playground}/{repo_to_top_folder[repo_name]}..."
        )
        subprocess.run(
            [
                "git",
                "clone",
                f"https://github.com/{repo_name}.git",
                f"{repo_playground}/{repo_to_top_folder[repo_name]}",
            ],
            check=True,
        )
        print("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        

def checkout_commit(repo_path, patch_path):
    """Checkout the specified commit in the given local git repository.
    :param repo_path: Path to the local git repository
    :param path_diff: path to git diff
    :return: None
    """
    try:
        # Change directory to the provided repository path and checkout the specified commit
        print(f"Checking out commit {patch_path} in repository at {repo_path}...")
        subprocess.run(["git", "-C", repo_path, "checkout", patch_path], check=True)
        print("Commit checked out successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def apply_patch(repo_path, diff_path):
    '''apply .diff to the origin code'''
    try:
        # Change directory to the provided repository path and checkout the specified commit
        print(f"Apply patch in repository at {diff_path}...")
        subprocess.run(["git", "-C", repo_path, "apply", diff_path], check=True)
        print("Patch applied successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def get_patch_commit(p_id):
    print(p_id)
    ds = load_dataset("princeton-nlp/SWE-bench", split="test",  cache_dir="/home/shared/huggingface")
    for i in range(len(ds)):
        if ds[i]['instance_id'] == p_id:
            commit_id = ds[i]['base_commit']
            patch = ds[i]['patch']
            repoName = ds[i]['repo']
            return commit_id, patch, repoName
        
