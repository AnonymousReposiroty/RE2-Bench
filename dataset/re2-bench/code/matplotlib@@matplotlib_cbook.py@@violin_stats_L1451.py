import numpy as np
from matplotlib import _api, _c_internal_utils, mlab

def violin_stats(X, method=("GaussianKDE", "scott"), points=100, quantiles=None):
    """
    Return a list of dictionaries of data which can be used to draw a series
    of violin plots.

    See the ``Returns`` section below to view the required keys of the
    dictionary.

    Users can skip this function and pass a user-defined set of dictionaries
    with the same keys to `~.axes.Axes.violin` instead of using Matplotlib
    to do the calculations. See the *Returns* section below for the keys
    that must be present in the dictionaries.

    Parameters
    ----------
    X : array-like
        Sample data that will be used to produce the gaussian kernel density
        estimates. Must have 2 or fewer dimensions.

    method : (name, bw_method) or callable,
        The method used to calculate the kernel density estimate for each
        column of data. Valid values:

        - a tuple of the form ``(name, bw_method)`` where *name* currently must
          always be ``"GaussianKDE"`` and *bw_method* is the method used to
          calculate the estimator bandwidth. Supported values are 'scott',
          'silverman' or a float or a callable. If a float, this will be used
          directly as `!kde.factor`.  If a callable, it should take a
          `matplotlib.mlab.GaussianKDE` instance as its only parameter and
          return a float.

        - a callable with the signature ::

             def method(data: ndarray, coords: ndarray) -> ndarray

          It should return the KDE of *data* evaluated at *coords*.

          .. versionadded:: 3.11
             Support for ``(name, bw_method)`` tuple.

    points : int, default: 100
        Defines the number of points to evaluate each of the gaussian kernel
        density estimates at.

    quantiles : array-like, default: None
        Defines (if not None) a list of floats in interval [0, 1] for each
        column of data, which represents the quantiles that will be rendered
        for that column of data. Must have 2 or fewer dimensions. 1D array will
        be treated as a singleton list containing them.

    Returns
    -------
    list of dict
        A list of dictionaries containing the results for each column of data.
        The dictionaries contain at least the following:

        - coords: A list of scalars containing the coordinates this particular
          kernel density estimate was evaluated at.
        - vals: A list of scalars containing the values of the kernel density
          estimate at each of the coordinates given in *coords*.
        - mean: The mean value for this column of data.
        - median: The median value for this column of data.
        - min: The minimum value for this column of data.
        - max: The maximum value for this column of data.
        - quantiles: The quantile values for this column of data.
    """
    if isinstance(method, tuple):
        name, bw_method = method
        if name != "GaussianKDE":
            raise ValueError(f"Unknown KDE method name {name!r}. The only supported "
                             'named method is "GaussianKDE"')

        def _kde_method(x, coords):
            # fallback gracefully if the vector contains only one value
            if np.all(x[0] == x):
                return (x[0] == coords).astype(float)
            kde = mlab.GaussianKDE(x, bw_method)
            return kde.evaluate(coords)

        method = _kde_method

    # List of dictionaries describing each of the violins.
    vpstats = []

    # Want X to be a list of data sequences
    X = _reshape_2D(X, "X")

    # Want quantiles to be as the same shape as data sequences
    if quantiles is not None and len(quantiles) != 0:
        quantiles = _reshape_2D(quantiles, "quantiles")
    # Else, mock quantiles if it's none or empty
    else:
        quantiles = [[]] * len(X)

    # quantiles should have the same size as dataset
    if len(X) != len(quantiles):
        raise ValueError("List of violinplot statistics and quantiles values"
                         " must have the same length")

    # Zip x and quantiles
    for (x, q) in zip(X, quantiles):
        # Dictionary of results for this distribution
        stats = {}

        # Calculate basic stats for the distribution
        min_val = np.min(x)
        max_val = np.max(x)
        quantile_val = np.percentile(x, 100 * q)

        # Evaluate the kernel density estimate
        coords = np.linspace(min_val, max_val, points)
        stats['vals'] = method(x, coords)
        stats['coords'] = coords

        # Store additional statistics for this distribution
        stats['mean'] = np.mean(x)
        stats['median'] = np.median(x)
        stats['min'] = min_val
        stats['max'] = max_val
        stats['quantiles'] = np.atleast_1d(quantile_val)

        # Append to output
        vpstats.append(stats)

    return vpstats
