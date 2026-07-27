
CLONR_DIR=$1
COMMIT_ID=$2
REPO_NAME=$3
REPO=$4
python3 -m venv .venv
source .venv/bin/activate

echo "Cloning Django repository into $CLONR_DIR/$REPO"
git clone https://github.com/$REPO_NAME.git $CLONR_DIR/$REPO
cd $CLONR_DIR/$REPO

echo "Checking out $COMMIT_ID"
git checkout $COMMIT_ID

echo "Applying patch"
git apply ../../patch.diff



python -m pip install --upgrade pip setuptools wheel
pip install -e .

pip install numpy
pip install sympy