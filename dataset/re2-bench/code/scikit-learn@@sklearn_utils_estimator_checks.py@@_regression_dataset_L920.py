from sklearn.datasets import (
    load_iris,
    make_blobs,
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.preprocessing import StandardScaler, scale

def _regression_dataset():
    global REGRESSION_DATASET
    if REGRESSION_DATASET is None:
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=1,
            bias=5.0,
            noise=20,
            random_state=42,
        )
        X = StandardScaler().fit_transform(X)
        REGRESSION_DATASET = X, y
    return REGRESSION_DATASET
