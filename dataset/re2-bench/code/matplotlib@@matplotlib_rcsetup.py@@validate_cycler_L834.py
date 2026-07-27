import ast
from cycler import Cycler, cycler as ccycler

def validate_cycler(s):
    """Return a Cycler object from a string repr or the object itself."""
    if isinstance(s, str):
        # TODO: We might want to rethink this...
        # While I think I have it quite locked down, it is execution of
        # arbitrary code without sanitation.
        # Combine this with the possibility that rcparams might come from the
        # internet (future plans), this could be downright dangerous.
        # I locked it down by only having the 'cycler()' function available.
        # UPDATE: Partly plugging a security hole.
        # I really should have read this:
        # https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
        # We should replace this eval with a combo of PyParsing and
        # ast.literal_eval()
        try:
            _DunderChecker().visit(ast.parse(s))
            s = eval(s, {'cycler': cycler, '__builtins__': {}})
        except BaseException as e:
            raise ValueError(f"{s!r} is not a valid cycler construction: {e}"
                             ) from e
    # Should make sure what comes from the above eval()
    # is a Cycler object.
    if isinstance(s, Cycler):
        cycler_inst = s
    else:
        raise ValueError(f"Object is not a string or Cycler instance: {s!r}")

    unknowns = cycler_inst.keys - (set(_prop_validators) | set(_prop_aliases))
    if unknowns:
        raise ValueError("Unknown artist properties: %s" % unknowns)

    # Not a full validation, but it'll at least normalize property names
    # A fuller validation would require v0.10 of cycler.
    checker = set()
    for prop in cycler_inst.keys:
        norm_prop = _prop_aliases.get(prop, prop)
        if norm_prop != prop and norm_prop in cycler_inst.keys:
            raise ValueError(f"Cannot specify both {norm_prop!r} and alias "
                             f"{prop!r} in the same prop_cycle")
        if norm_prop in checker:
            raise ValueError(f"Another property was already aliased to "
                             f"{norm_prop!r}. Collision normalizing {prop!r}.")
        checker.update([norm_prop])

    # This is just an extra-careful check, just in case there is some
    # edge-case I haven't thought of.
    assert len(checker) == len(cycler_inst.keys)

    # Now, it should be safe to mutate this cycler
    for prop in cycler_inst.keys:
        norm_prop = _prop_aliases.get(prop, prop)
        cycler_inst.change_key(prop, norm_prop)

    for key, vals in cycler_inst.by_key().items():
        _prop_validators[key](vals)

    return cycler_inst
