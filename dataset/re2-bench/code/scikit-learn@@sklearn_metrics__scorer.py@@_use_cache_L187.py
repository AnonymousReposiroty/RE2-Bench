from collections import Counter
from sklearn.utils.validation import _check_response_method

class _MultimetricScorer:
    """Callable for multimetric scoring used to avoid repeated calls
    to `predict_proba`, `predict`, and `decision_function`.

    `_MultimetricScorer` will return a dictionary of scores corresponding to
    the scorers in the dictionary. Note that `_MultimetricScorer` can be
    created with a dictionary with one key  (i.e. only one actual scorer).

    Parameters
    ----------
    scorers : dict
        Dictionary mapping names to callable scorers.

    raise_exc : bool, default=True
        Whether to raise the exception in `__call__` or not. If set to `False`
        a formatted string of the exception details is passed as result of
        the failing scorer.
    """

    def __init__(self, *, scorers, raise_exc=True):
        self._scorers = scorers
        self._raise_exc = raise_exc

    def _use_cache(self, estimator):
        """Return True if using a cache is beneficial, thus when a response method will
        be called several time.
        """
        if len(self._scorers) == 1:
            return False
        counter = Counter([_check_response_method(estimator, scorer._response_method).__name__ for scorer in self._scorers.values() if isinstance(scorer, _BaseScorer)])
        if any((val > 1 for val in counter.values())):
            return True
        return False
