CLONR_DIR=$1
COMMIT_ID=$2
REPO_NAME=$3
REPO=$4


python3.8 -m venv .venv
source .venv/bin/activate

python -m pip install -U "pip<24" "setuptools<60" "wheel<0.38"


pip install "numpy==1.21.6" "scipy==1.7.3" "cython==0.29.33" joblib==1.1.1 threadpoolctl==3.1.0
pip install sympy

echo "Cloning Scikit-learn repository into $CLONR_DIR/$REPO"
git clone https://github.com/$REPO_NAME.git $CLONR_DIR/$REPO
cd $CLONR_DIR/$REPO

echo Checking out $COMMIT_ID
git checkout $COMMIT_ID

git apply ../../patch.diff

git clean -xdf
pip install . --no-build-isolation -v
