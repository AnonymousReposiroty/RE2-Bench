import operator
from collections.abc import Iterable, Mapping, Sequence
from functools import partial, reduce
from itertools import product
import numpy as np

class ParameterGrid:
    """Grid of parameters with a discrete number of values for each.

    Can be used to iterate over parameter value combinations with the
    Python built-in function iter.
    The order of the generated parameter combinations is deterministic.

    Read more in the :ref:`User Guide <grid_search>`.

    Parameters
    ----------
    param_grid : dict of str to sequence, or sequence of such
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.

        An empty dict signifies default parameters.

        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See the examples below.

    Examples
    --------
    >>> from sklearn.model_selection import ParameterGrid
    >>> param_grid = {'a': [1, 2], 'b': [True, False]}
    >>> list(ParameterGrid(param_grid)) == (
    ...    [{'a': 1, 'b': True}, {'a': 1, 'b': False},
    ...     {'a': 2, 'b': True}, {'a': 2, 'b': False}])
    True

    >>> grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
    >>> list(ParameterGrid(grid)) == [{'kernel': 'linear'},
    ...                               {'kernel': 'rbf', 'gamma': 1},
    ...                               {'kernel': 'rbf', 'gamma': 10}]
    True
    >>> ParameterGrid(grid)[1] == {'kernel': 'rbf', 'gamma': 1}
    True

    See Also
    --------
    GridSearchCV : Uses :class:`ParameterGrid` to perform a full parallelized
        parameter search.
    """

    def __init__(self, param_grid):
        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError(f'Parameter grid should be a dict or a list, got: {param_grid!r} of type {type(param_grid).__name__}')
        if isinstance(param_grid, Mapping):
            param_grid = [param_grid]
        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError(f'Parameter grid is not a dict ({grid!r})')
            for key, value in grid.items():
                if isinstance(value, np.ndarray) and value.ndim > 1:
                    raise ValueError(f'Parameter array for {key!r} should be one-dimensional, got: {value!r} with shape {value.shape}')
                if isinstance(value, str) or not isinstance(value, (np.ndarray, Sequence)):
                    raise TypeError(f'Parameter grid for parameter {key!r} needs to be a list or a numpy array, but got {value!r} (of type {type(value).__name__}) instead. Single values need to be wrapped in a list with one element.')
                if len(value) == 0:
                    raise ValueError(f'Parameter grid for parameter {key!r} need to be a non-empty sequence, got: {value!r}')
        self.param_grid = param_grid

    def __len__(self):
        """Number of points on the grid."""
        product = partial(reduce, operator.mul)
        return sum((product((len(v) for v in p.values())) if p else 1 for p in self.param_grid))
