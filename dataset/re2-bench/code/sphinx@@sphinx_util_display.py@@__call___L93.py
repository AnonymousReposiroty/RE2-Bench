import functools
from collections.abc import Callable, Iterable, Iterator

class progress_message:

    def __init__(self, message: str, *, nonl: bool=True) -> None:
        self.message = message
        self.nonl = nonl

    def __call__(self, f: Callable[P, R]) -> Callable[P, R]:

        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with self:
                return f(*args, **kwargs)
        return wrapper
