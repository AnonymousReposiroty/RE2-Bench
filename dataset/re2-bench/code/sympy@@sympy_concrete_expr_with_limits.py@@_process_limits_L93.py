from sympy.core.containers import Tuple
from sympy.core.relational import Equality, Relational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.logic.boolalg import BooleanFunction
from sympy.sets.sets import Interval, Set
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import Idx
from sympy.utilities import flatten
from sympy.utilities.iterables import sift, is_sequence

def _process_limits(*symbols, discrete=None):
    """Process the list of symbols and convert them to canonical limits,
    storing them as Tuple(symbol, lower, upper). The orientation of
    the function is also returned when the upper limit is missing
    so (x, 1, None) becomes (x, None, 1) and the orientation is changed.
    In the case that a limit is specified as (symbol, Range), a list of
    length 4 may be returned if a change of variables is needed; the
    expression that should replace the symbol in the expression is
    the fourth element in the list.
    """
    limits = []
    orientation = 1
    if discrete is None:
        err_msg = 'discrete must be True or False'
    elif discrete:
        err_msg = 'use Range, not Interval or Relational'
    else:
        err_msg = 'use Interval or Relational, not Range'
    for V in symbols:
        if isinstance(V, (Relational, BooleanFunction)):
            if discrete:
                raise TypeError(err_msg)
            variable = V.atoms(Symbol).pop()
            V = (variable, V.as_set())
        elif isinstance(V, Symbol) or getattr(V, '_diff_wrt', False):
            if isinstance(V, Idx):
                if V.lower is None or V.upper is None:
                    limits.append(Tuple(V))
                else:
                    limits.append(Tuple(V, V.lower, V.upper))
            else:
                limits.append(Tuple(V))
            continue
        if is_sequence(V) and not isinstance(V, Set):
            if len(V) == 2 and isinstance(V[1], Set):
                V = list(V)
                if isinstance(V[1], Interval):  # includes Reals
                    if discrete:
                        raise TypeError(err_msg)
                    V[1:] = V[1].inf, V[1].sup
                elif isinstance(V[1], Range):
                    if not discrete:
                        raise TypeError(err_msg)
                    lo = V[1].inf
                    hi = V[1].sup
                    dx = abs(V[1].step)  # direction doesn't matter
                    if dx == 1:
                        V[1:] = [lo, hi]
                    else:
                        if lo is not S.NegativeInfinity:
                            V = [V[0]] + [0, (hi - lo)//dx, dx*V[0] + lo]
                        else:
                            V = [V[0]] + [0, S.Infinity, -dx*V[0] + hi]
                else:
                    # more complicated sets would require splitting, e.g.
                    # Union(Interval(1, 3), interval(6,10))
                    raise NotImplementedError(
                        'expecting Range' if discrete else
                        'Relational or single Interval' )
            V = sympify(flatten(V))  # list of sympified elements/None
            if isinstance(V[0], (Symbol, Idx)) or getattr(V[0], '_diff_wrt', False):
                newsymbol = V[0]
                if len(V) == 3:
                    # general case
                    if V[2] is None and V[1] is not None:
                        orientation *= -1
                    V = [newsymbol] + [i for i in V[1:] if i is not None]

                lenV = len(V)
                if not isinstance(newsymbol, Idx) or lenV == 3:
                    if lenV == 4:
                        limits.append(Tuple(*V))
                        continue
                    if lenV == 3:
                        if isinstance(newsymbol, Idx):
                            # Idx represents an integer which may have
                            # specified values it can take on; if it is
                            # given such a value, an error is raised here
                            # if the summation would try to give it a larger
                            # or smaller value than permitted. None and Symbolic
                            # values will not raise an error.
                            lo, hi = newsymbol.lower, newsymbol.upper
                            try:
                                if lo is not None and not bool(V[1] >= lo):
                                    raise ValueError("Summation will set Idx value too low.")
                            except TypeError:
                                pass
                            try:
                                if hi is not None and not bool(V[2] <= hi):
                                    raise ValueError("Summation will set Idx value too high.")
                            except TypeError:
                                pass
                        limits.append(Tuple(*V))
                        continue
                    if lenV == 1 or (lenV == 2 and V[1] is None):
                        limits.append(Tuple(newsymbol))
                        continue
                    elif lenV == 2:
                        limits.append(Tuple(newsymbol, V[1]))
                        continue

        raise ValueError('Invalid limits given: %s' % str(symbols))

    return limits, orientation
