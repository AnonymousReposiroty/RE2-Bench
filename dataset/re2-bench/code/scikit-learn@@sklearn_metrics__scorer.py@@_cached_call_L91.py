from sklearn.utils._response import _get_response_values

def _cached_call(cache, estimator, response_method, *args, **kwargs):
    """Call estimator with method and args and kwargs."""
    if cache is not None and response_method in cache:
        return cache[response_method]

    result, _ = _get_response_values(
        estimator, *args, response_method=response_method, **kwargs
    )

    if cache is not None:
        cache[response_method] = result

    return result
