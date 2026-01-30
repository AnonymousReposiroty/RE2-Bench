from sklearn.metrics.cluster import mutual_info_score

def _compute_mi(x, y, x_discrete, y_discrete, n_neighbors=3):
    """Compute mutual information between two variables.

    This is a simple wrapper which selects a proper function to call based on
    whether `x` and `y` are discrete or not.
    """
    if x_discrete and y_discrete:
        return mutual_info_score(x, y)
    elif x_discrete and not y_discrete:
        return _compute_mi_cd(y, x, n_neighbors)
    elif not x_discrete and y_discrete:
        return _compute_mi_cd(x, y, n_neighbors)
    else:
        return _compute_mi_cc(x, y, n_neighbors)
