def _check_gcv_mode(X, gcv_mode):
    if gcv_mode in ["eigen", "svd"]:
        return gcv_mode
    # if X has more rows than columns, use decomposition of X^T.X,
    # otherwise X.X^T
    if X.shape[0] > X.shape[1]:
        return "svd"
    return "eigen"
