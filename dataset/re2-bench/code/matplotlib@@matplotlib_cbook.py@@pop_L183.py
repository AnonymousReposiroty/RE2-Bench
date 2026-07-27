class _UnhashDict:
    """
    A minimal dict-like class that also supports unhashable keys, storing them
    in a list of key-value pairs.

    This class only implements the interface needed for `CallbackRegistry`, and
    tries to minimize the overhead for the hashable case.
    """

    def __init__(self, pairs):
        self._dict = {}
        self._pairs = []
        for k, v in pairs:
            self[k] = v

    def pop(self, key, *args):
        try:
            if key in self._dict:
                return self._dict.pop(key)
        except TypeError:
            for i, (k, v) in enumerate(self._pairs):
                if k == key:
                    del self._pairs[i]
                    return v
        if args:
            return args[0]
        raise KeyError(key)
