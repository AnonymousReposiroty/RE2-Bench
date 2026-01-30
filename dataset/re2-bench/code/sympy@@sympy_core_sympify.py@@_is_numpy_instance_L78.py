def _is_numpy_instance(a):
    """
    Checks if an object is an instance of a type from the numpy module.
    """
    # This check avoids unnecessarily importing NumPy.  We check the whole
    # __mro__ in case any base type is a numpy type.
    return any(type_.__module__ == 'numpy'
               for type_ in type(a).__mro__)
