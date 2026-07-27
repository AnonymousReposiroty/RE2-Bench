import os
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.error import HTTPError, URLError
from warnings import warn

def _retry_with_clean_cache(
    openml_path: str,
    data_home: Optional[str],
    no_retry_exception: Optional[Exception] = None,
) -> Callable:
    """If the first call to the decorated function fails, the local cached
    file is removed, and the function is called again. If ``data_home`` is
    ``None``, then the function is called once. We can provide a specific
    exception to not retry on using `no_retry_exception` parameter.
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kw):
            if data_home is None:
                return f(*args, **kw)
            try:
                return f(*args, **kw)
            except URLError:
                raise
            except Exception as exc:
                if no_retry_exception is not None and isinstance(
                    exc, no_retry_exception
                ):
                    raise
                warn("Invalid cache, redownloading file", RuntimeWarning)
                local_path = _get_local_path(openml_path, data_home)
                if os.path.exists(local_path):
                    os.unlink(local_path)
                return f(*args, **kw)

        return wrapper

    return decorator
