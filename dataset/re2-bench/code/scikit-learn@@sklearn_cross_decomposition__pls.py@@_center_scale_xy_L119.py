import numpy as np

def _center_scale_xy(X, y, scale=True):
    """Center X, y and scale if the scale parameter==True

    Returns
    -------
        X, y, x_mean, y_mean, x_std, y_std
    """
    # center
    x_mean = X.mean(axis=0)
    X -= x_mean
    y_mean = y.mean(axis=0)
    y -= y_mean
    # scale
    if scale:
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        y /= y_std
    else:
        x_std = np.ones(X.shape[1])
        y_std = np.ones(y.shape[1])
    return X, y, x_mean, y_mean, x_std, y_std
