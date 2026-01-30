def precedence(item):
    """Returns the precedence of a given object.

    This is the precedence for StrPrinter.
    """
    if hasattr(item, "precedence"):
        return item.precedence
    if not isinstance(item, type):
        for i in type(item).mro():
            n = i.__name__
            if n in PRECEDENCE_FUNCTIONS:
                return PRECEDENCE_FUNCTIONS[n](item)
            elif n in PRECEDENCE_VALUES:
                return PRECEDENCE_VALUES[n]
    return PRECEDENCE["Atom"]
