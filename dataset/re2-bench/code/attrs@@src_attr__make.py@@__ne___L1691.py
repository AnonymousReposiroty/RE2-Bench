def __ne__(self, other):
    """
    Check equality and either forward a NotImplemented or
    return the result negated.
    """
    result = self.__eq__(other)
    if result is NotImplemented:
        return NotImplemented

    return not result
