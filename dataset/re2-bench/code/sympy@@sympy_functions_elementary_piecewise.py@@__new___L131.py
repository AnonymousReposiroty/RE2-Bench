from sympy.core.basic import Basic, as_Basic
from sympy.core.function import DefinedFunction
from sympy.core.parameters import global_parameters
from sympy.logic.boolalg import (And, Boolean, distribute_and_over_or, Not,
    true, false, Or, ITE, simplify_logic, to_cnf, distribute_or_over_and)
from sympy.utilities.misc import filldedent, func_name

class Piecewise(DefinedFunction):
    """
    Represents a piecewise function.

    Usage:

      Piecewise( (expr,cond), (expr,cond), ... )
        - Each argument is a 2-tuple defining an expression and condition
        - The conds are evaluated in turn returning the first that is True.
          If any of the evaluated conds are not explicitly False,
          e.g. ``x < 1``, the function is returned in symbolic form.
        - If the function is evaluated at a place where all conditions are False,
          nan will be returned.
        - Pairs where the cond is explicitly False, will be removed and no pair
          appearing after a True condition will ever be retained. If a single
          pair with a True condition remains, it will be returned, even when
          evaluation is False.

    Examples
    ========

    >>> from sympy import Piecewise, log, piecewise_fold
    >>> from sympy.abc import x, y
    >>> f = x**2
    >>> g = log(x)
    >>> p = Piecewise((0, x < -1), (f, x <= 1), (g, True))
    >>> p.subs(x,1)
    1
    >>> p.subs(x,5)
    log(5)

    Booleans can contain Piecewise elements:

    >>> cond = (x < y).subs(x, Piecewise((2, x < 0), (3, True))); cond
    Piecewise((2, x < 0), (3, True)) < y

    The folded version of this results in a Piecewise whose
    expressions are Booleans:

    >>> folded_cond = piecewise_fold(cond); folded_cond
    Piecewise((2 < y, x < 0), (3 < y, True))

    When a Boolean containing Piecewise (like cond) or a Piecewise
    with Boolean expressions (like folded_cond) is used as a condition,
    it is converted to an equivalent :class:`~.ITE` object:

    >>> Piecewise((1, folded_cond))
    Piecewise((1, ITE(x < 0, y > 2, y > 3)))

    When a condition is an ``ITE``, it will be converted to a simplified
    Boolean expression:

    >>> piecewise_fold(_)
    Piecewise((1, ((x >= 0) | (y > 2)) & ((y > 3) | (x < 0))))

    See Also
    ========

    piecewise_fold
    piecewise_exclusive
    ITE
    """
    nargs = None
    is_Piecewise = True

    def __new__(cls, *args, **options):
        if len(args) == 0:
            raise TypeError('At least one (expr, cond) pair expected.')
        newargs = []
        for ec in args:
            pair = ExprCondPair(*getattr(ec, 'args', ec))
            cond = pair.cond
            if cond is false:
                continue
            newargs.append(pair)
            if cond is true:
                break
        eval = options.pop('evaluate', global_parameters.evaluate)
        if eval:
            r = cls.eval(*newargs)
            if r is not None:
                return r
        elif len(newargs) == 1 and newargs[0].cond == True:
            return newargs[0].expr
        return Basic.__new__(cls, *newargs, **options)

    @classmethod
    def eval(cls, *_args):
        """Either return a modified version of the args or, if no
        modifications were made, return None.

        Modifications that are made here:

        1. relationals are made canonical
        2. any False conditions are dropped
        3. any repeat of a previous condition is ignored
        4. any args past one with a true condition are dropped

        If there are no args left, nan will be returned.
        If there is a single arg with a True condition, its
        corresponding expression will be returned.

        EXAMPLES
        ========

        >>> from sympy import Piecewise
        >>> from sympy.abc import x
        >>> cond = -x < -1
        >>> args = [(1, cond), (4, cond), (3, False), (2, True), (5, x < 1)]
        >>> Piecewise(*args, evaluate=False)
        Piecewise((1, -x < -1), (4, -x < -1), (2, True))
        >>> Piecewise(*args)
        Piecewise((1, x > 1), (2, True))
        """
        if not _args:
            return Undefined
        if len(_args) == 1 and _args[0][-1] == True:
            return _args[0][0]
        newargs = _piecewise_collapse_arguments(_args)
        missing = len(newargs) != len(_args)
        same = all((a == b for a, b in zip(newargs, _args)))
        if not newargs:
            raise ValueError(filldedent('\n                There are no conditions (or none that\n                are not trivially false) to define an\n                expression.'))
        if missing or not same:
            return cls(*newargs)
    _eval_is_finite = lambda self: self._eval_template_is_attr('is_finite')
    _eval_is_complex = lambda self: self._eval_template_is_attr('is_complex')
    _eval_is_even = lambda self: self._eval_template_is_attr('is_even')
    _eval_is_imaginary = lambda self: self._eval_template_is_attr('is_imaginary')
    _eval_is_integer = lambda self: self._eval_template_is_attr('is_integer')
    _eval_is_irrational = lambda self: self._eval_template_is_attr('is_irrational')
    _eval_is_negative = lambda self: self._eval_template_is_attr('is_negative')
    _eval_is_nonnegative = lambda self: self._eval_template_is_attr('is_nonnegative')
    _eval_is_nonpositive = lambda self: self._eval_template_is_attr('is_nonpositive')
    _eval_is_nonzero = lambda self: self._eval_template_is_attr('is_nonzero')
    _eval_is_odd = lambda self: self._eval_template_is_attr('is_odd')
    _eval_is_polar = lambda self: self._eval_template_is_attr('is_polar')
    _eval_is_positive = lambda self: self._eval_template_is_attr('is_positive')
    _eval_is_extended_real = lambda self: self._eval_template_is_attr('is_extended_real')
    _eval_is_extended_positive = lambda self: self._eval_template_is_attr('is_extended_positive')
    _eval_is_extended_negative = lambda self: self._eval_template_is_attr('is_extended_negative')
    _eval_is_extended_nonzero = lambda self: self._eval_template_is_attr('is_extended_nonzero')
    _eval_is_extended_nonpositive = lambda self: self._eval_template_is_attr('is_extended_nonpositive')
    _eval_is_extended_nonnegative = lambda self: self._eval_template_is_attr('is_extended_nonnegative')
    _eval_is_real = lambda self: self._eval_template_is_attr('is_real')
    _eval_is_zero = lambda self: self._eval_template_is_attr('is_zero')
