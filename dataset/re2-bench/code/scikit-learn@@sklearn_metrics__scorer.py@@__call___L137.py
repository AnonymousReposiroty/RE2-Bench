from collections import Counter
from functools import partial
from traceback import format_exc
from sklearn.utils import Bunch
from sklearn.utils.metadata_routing import (
    MetadataRequest,
    MetadataRouter,
    MethodMapping,
    _MetadataRequester,
    _raise_for_params,
    _routing_enabled,
    get_routing_for_object,
    process_routing,
)
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

    def __call__(self, estimator, *args, **kwargs):
        """Evaluate predicted target values."""
        scores = {}
        cache = {} if self._use_cache(estimator) else None
        cached_call = partial(_cached_call, cache)
        if _routing_enabled():
            routed_params = process_routing(self, 'score', **kwargs)
        else:
            common_kwargs = {arg: value for arg, value in kwargs.items() if arg != 'sample_weight'}
            routed_params = Bunch(**{name: Bunch(score=common_kwargs.copy()) for name in self._scorers})
            if 'sample_weight' in kwargs:
                for name, scorer in self._scorers.items():
                    if scorer._accept_sample_weight():
                        routed_params[name].score['sample_weight'] = kwargs['sample_weight']
        for name, scorer in self._scorers.items():
            try:
                if isinstance(scorer, _BaseScorer):
                    score = scorer._score(cached_call, estimator, *args, **routed_params.get(name).score)
                else:
                    score = scorer(estimator, *args, **routed_params.get(name).score)
                scores[name] = score
            except Exception as e:
                if self._raise_exc:
                    raise e
                else:
                    scores[name] = format_exc()
        return scores

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
