from typing import TypeVar, Iterable, Callable

def _sift_true_false(seq: Iterable[T], keyfunc: Callable[[T], bool]) -> tuple[list[T], list[T]]:
    """Sift iterable for items with keyfunc(item) = True/False."""
    true: list[T] = []
    false: list[T] = []
    for i in seq:
        if keyfunc(i):
            true.append(i)
        else:
            false.append(i)
    return true, false
