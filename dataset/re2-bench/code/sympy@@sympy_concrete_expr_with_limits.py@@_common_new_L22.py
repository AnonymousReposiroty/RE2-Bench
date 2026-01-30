from sympy.core.containers import Tuple
from sympy.core.relational import Equality, Relational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.functions.elementary.piecewise import (piecewise_fold,
    Piecewise)
from sympy.utilities.exceptions import sympy_deprecation_warning

def _common_new(cls, function, *symbols, discrete, **assumptions):
    """Return either a special return value or the tuple,
    (function, limits, orientation). This code is common to
    both ExprWithLimits and AddWithLimits."""
    function = sympify(function)

    if isinstance(function, Equality):
        # This transforms e.g. Integral(Eq(x, y)) to Eq(Integral(x), Integral(y))
        # but that is only valid for definite integrals.
        limits, orientation = _process_limits(*symbols, discrete=discrete)
        if not (limits and all(len(limit) == 3 for limit in limits)):
            sympy_deprecation_warning(
                """
                Creating a indefinite integral with an Eq() argument is
                deprecated.

                This is because indefinite integrals do not preserve equality
                due to the arbitrary constants. If you want an equality of
                indefinite integrals, use Eq(Integral(a, x), Integral(b, x))
                explicitly.
                """,
                deprecated_since_version="1.6",
                active_deprecations_target="deprecated-indefinite-integral-eq",
                stacklevel=5,
            )

        lhs = function.lhs
        rhs = function.rhs
        return Equality(cls(lhs, *symbols, **assumptions), \
                        cls(rhs, *symbols, **assumptions))

    if function is S.NaN:
        return S.NaN

    if symbols:
        limits, orientation = _process_limits(*symbols, discrete=discrete)
        for i, li in enumerate(limits):
            if len(li) == 4:
                function = function.subs(li[0], li[-1])
                limits[i] = Tuple(*li[:-1])
    else:
        # symbol not provided -- we can still try to compute a general form
        free = function.free_symbols
        if len(free) != 1:
            raise ValueError(
                "specify dummy variables for %s" % function)
        limits, orientation = [Tuple(s) for s in free], 1

    # denest any nested calls
    while cls == type(function):
        limits = list(function.limits) + limits
        function = function.function

    # Any embedded piecewise functions need to be brought out to the
    # top level. We only fold Piecewise that contain the integration
    # variable.
    reps = {}
    symbols_of_integration = {i[0] for i in limits}
    for p in function.atoms(Piecewise):
        if not p.has(*symbols_of_integration):
            reps[p] = Dummy()
    # mask off those that don't
    function = function.xreplace(reps)
    # do the fold
    function = piecewise_fold(function)
    # remove the masking
    function = function.xreplace({v: k for k, v in reps.items()})

    return function, limits, orientation
