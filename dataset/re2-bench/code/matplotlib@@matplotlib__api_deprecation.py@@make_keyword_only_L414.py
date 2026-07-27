import functools
import inspect

def make_keyword_only(since, name, func=None):
    """
    Decorator indicating that passing parameter *name* (or any of the following
    ones) positionally to *func* is being deprecated.

    When used on a method that has a pyplot wrapper, this should be the
    outermost decorator, so that :file:`boilerplate.py` can access the original
    signature.

    Examples
    --------
    Assume we want to only allow *dataset* and *positions* as positional
    parameters on the method ::

        def violinplot(self, dataset, positions=None, vert=None, ...)

    Introduce the deprecation by adding the decorator ::

        @_api.make_keyword_only("3.10", "vert")
        def violinplot(self, dataset, positions=None, vert=None, ...)

    When the deprecation expires, switch to ::

        def violinplot(self, dataset, positions=None, *, vert=None, ...)

    """

    decorator = functools.partial(make_keyword_only, since, name)

    if func is None:
        return decorator

    signature = inspect.signature(func)
    POK = inspect.Parameter.POSITIONAL_OR_KEYWORD
    KWO = inspect.Parameter.KEYWORD_ONLY
    assert (name in signature.parameters
            and signature.parameters[name].kind == POK), (
        f"Matplotlib internal error: {name!r} must be a positional-or-keyword "
        f"parameter for {func.__name__}(). If this error happens on a function with a "
        f"pyplot wrapper, make sure make_keyword_only() is the outermost decorator.")
    names = [*signature.parameters]
    name_idx = names.index(name)
    kwonly = [name for name in names[name_idx:]
              if signature.parameters[name].kind == POK]

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Don't use signature.bind here, as it would fail when stacked with
        # rename_parameter and an "old" argument name is passed in
        # (signature.bind would fail, but the actual call would succeed).
        if len(args) > name_idx:
            warn_deprecated(
                since, message="Passing the %(name)s %(obj_type)s "
                "positionally is deprecated since Matplotlib %(since)s; the "
                "parameter will become keyword-only in %(removal)s.",
                name=name, obj_type=f"parameter of {func.__name__}()")
        return func(*args, **kwargs)

    # Don't modify *func*'s signature, as boilerplate.py needs it.
    wrapper.__signature__ = signature.replace(parameters=[
        param.replace(kind=KWO) if param.name in kwonly else param
        for param in signature.parameters.values()])
    DECORATORS[wrapper] = decorator
    return wrapper
