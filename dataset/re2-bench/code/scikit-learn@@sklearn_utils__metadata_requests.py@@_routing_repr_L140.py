def _routing_repr(obj):
    """Get a representation suitable for messages printed in the routing machinery.

    This is different than `repr(obj)`, since repr(estimator) can be verbose when
    there are many constructor arguments set by the user.

    This is most suitable for Scorers as it gives a nice representation of what they
    are. This is done by implementing a `_routing_repr` method on the object.

    Since the `owner` object could be the type name (str), we return that string if the
    given `obj` is a string, otherwise we return the object's type name.

    .. versionadded:: 1.8
    """
    try:
        return obj._routing_repr()
    except AttributeError:
        return obj if isinstance(obj, str) else type(obj).__name__
