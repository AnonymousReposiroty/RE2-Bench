import warnings
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

class _BaseScorer(_MetadataRequester):
    """Base scorer that is used as `scorer(estimator, X, y_true)`.

    Parameters
    ----------
    score_func : callable
        The score function to use. It will be called as
        `score_func(y_true, y_pred, **kwargs)`.

    sign : int
        Either 1 or -1 to returns the score with `sign * score_func(estimator, X, y)`.
        Thus, `sign` defined if higher scores are better or worse.

    kwargs : dict
        Additional parameters to pass to the score function.

    response_method : str
        The method to call on the estimator to get the response values.
    """

    def __init__(self, score_func, sign, kwargs, response_method='predict'):
        self._score_func = score_func
        self._sign = sign
        self._kwargs = kwargs
        self._response_method = response_method

    def _warn_overlap(self, message, kwargs):
        """Warn if there is any overlap between ``self._kwargs`` and ``kwargs``.

        This method is intended to be used to check for overlap between
        ``self._kwargs`` and ``kwargs`` passed as metadata.
        """
        _kwargs = set() if self._kwargs is None else set(self._kwargs.keys())
        overlap = _kwargs.intersection(kwargs.keys())
        if overlap:
            warnings.warn(f'{message} Overlapping parameters are: {overlap}', UserWarning)

    def set_score_request(self, **kwargs):
        """Set requested parameters by the scorer.

        Please see :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.3

        Parameters
        ----------
        kwargs : dict
            Arguments should be of the form ``param_name=alias``, and `alias`
            can be one of ``{True, False, None, str}``.
        """
        if not _routing_enabled():
            raise RuntimeError('This method is only available when metadata routing is enabled. You can enable it using sklearn.set_config(enable_metadata_routing=True).')
        self._warn_overlap(message='You are setting metadata request for parameters which are already set as kwargs for this metric. These set values will be overridden by passed metadata if provided. Please pass them either as metadata or kwargs to `make_scorer`.', kwargs=kwargs)
        self._metadata_request = MetadataRequest(owner=self)
        for param, alias in kwargs.items():
            self._metadata_request.score.add_request(param=param, alias=alias)
        return self
