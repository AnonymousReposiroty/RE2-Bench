from sympy.core import (sympify, Basic, S, Expr, factor_terms,
                        Mul, Add, bottom_up)
from sympy.functions import sin, cos, exp, cosh, tanh, sinh, tan, cot, coth
from sympy.simplify.fu import fu

def trigsimp(expr, inverse=False, **opts):
    """Returns a reduced expression by using known trig identities.

    Parameters
    ==========

    inverse : bool, optional
        If ``inverse=True``, it will be assumed that a composition of inverse
        functions, such as sin and asin, can be cancelled in any order.
        For example, ``asin(sin(x))`` will yield ``x`` without checking whether
        x belongs to the set where this relation is true. The default is False.
        Default : True

    method : string, optional
        Specifies the method to use. Valid choices are:

        - ``'matching'``, default
        - ``'groebner'``
        - ``'combined'``
        - ``'fu'``
        - ``'old'``

        If ``'matching'``, simplify the expression recursively by targeting
        common patterns. If ``'groebner'``, apply an experimental groebner
        basis algorithm. In this case further options are forwarded to
        ``trigsimp_groebner``, please refer to
        its docstring. If ``'combined'``, it first runs the groebner basis
        algorithm with small default parameters, then runs the ``'matching'``
        algorithm. If ``'fu'``, run the collection of trigonometric
        transformations described by Fu, et al. (see the
        :py:func:`~sympy.simplify.fu.fu` docstring). If ``'old'``, the original
        SymPy trig simplification function is run.
    opts :
        Optional keyword arguments passed to the method. See each method's
        function docstring for details.

    Examples
    ========

    >>> from sympy import trigsimp, sin, cos, log
    >>> from sympy.abc import x
    >>> e = 2*sin(x)**2 + 2*cos(x)**2
    >>> trigsimp(e)
    2

    Simplification occurs wherever trigonometric functions are located.

    >>> trigsimp(log(e))
    log(2)

    Using ``method='groebner'`` (or ``method='combined'``) might lead to
    greater simplification.

    The old trigsimp routine can be accessed as with method ``method='old'``.

    >>> from sympy import coth, tanh
    >>> t = 3*tanh(x)**7 - 2/coth(x)**7
    >>> trigsimp(t, method='old') == t
    True
    >>> trigsimp(t)
    tanh(x)**7

    """
    from sympy.simplify.fu import fu

    expr = sympify(expr)

    _eval_trigsimp = getattr(expr, '_eval_trigsimp', None)
    if _eval_trigsimp is not None:
        return _eval_trigsimp(**opts)

    old = opts.pop('old', False)
    if not old:
        opts.pop('deep', None)
        opts.pop('recursive', None)
        method = opts.pop('method', 'matching')
    else:
        method = 'old'

    def groebnersimp(ex, **opts):
        def traverse(e):
            if e.is_Atom:
                return e
            args = [traverse(x) for x in e.args]
            if e.is_Function or e.is_Pow:
                args = [trigsimp_groebner(x, **opts) for x in args]
            return e.func(*args)
        new = traverse(ex)
        if not isinstance(new, Expr):
            return new
        return trigsimp_groebner(new, **opts)

    trigsimpfunc = {
        'fu': (lambda x: fu(x, **opts)),
        'matching': (lambda x: futrig(x)),
        'groebner': (lambda x: groebnersimp(x, **opts)),
        'combined': (lambda x: futrig(groebnersimp(x,
                               polynomial=True, hints=[2, tan]))),
        'old': lambda x: trigsimp_old(x, **opts),
                   }[method]

    expr_simplified = trigsimpfunc(expr)
    if inverse:
        expr_simplified = _trigsimp_inverse(expr_simplified)

    return expr_simplified
