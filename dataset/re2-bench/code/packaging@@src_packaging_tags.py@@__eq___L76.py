import platform

class Tag:
    """
    A representation of the tag triple for a wheel.

    Instances are considered immutable and thus are hashable. Equality checking
    is also supported.
    """
    __slots__ = ['_abi', '_hash', '_interpreter', '_platform']

    def __init__(self, interpreter: str, abi: str, platform: str) -> None:
        self._interpreter = interpreter.lower()
        self._abi = abi.lower()
        self._platform = platform.lower()
        self._hash = hash((self._interpreter, self._abi, self._platform))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tag):
            return NotImplemented
        return self._hash == other._hash and self._platform == other._platform and (self._abi == other._abi) and (self._interpreter == other._interpreter)
