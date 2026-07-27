from typing import TYPE_CHECKING, ClassVar, overload, Literal
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .cache import cacheit
from .logic import fuzzy_not, _fuzzy_group
from .expr import Expr
from .kind import KindDispatcher
from sympy.utilities.iterables import sift

class Mul(Expr, AssocOp):
    """
    Expression representing multiplication operation for algebraic field.

    .. deprecated:: 1.7

       Using arguments that aren't subclasses of :class:`~.Expr` in core
       operators (:class:`~.Mul`, :class:`~.Add`, and :class:`~.Pow`) is
       deprecated. See :ref:`non-expr-args-deprecated` for details.

    Every argument of ``Mul()`` must be ``Expr``. Infix operator ``*``
    on most scalar objects in SymPy calls this class.

    Another use of ``Mul()`` is to represent the structure of abstract
    multiplication so that its arguments can be substituted to return
    different class. Refer to examples section for this.

    ``Mul()`` evaluates the argument unless ``evaluate=False`` is passed.
    The evaluation logic includes:

    1. Flattening
        ``Mul(x, Mul(y, z))`` -> ``Mul(x, y, z)``

    2. Identity removing
        ``Mul(x, 1, y)`` -> ``Mul(x, y)``

    3. Exponent collecting by ``.as_base_exp()``
        ``Mul(x, x**2)`` -> ``Pow(x, 3)``

    4. Term sorting
        ``Mul(y, x, 2)`` -> ``Mul(2, x, y)``

    Since multiplication can be vector space operation, arguments may
    have the different :obj:`sympy.core.kind.Kind()`. Kind of the
    resulting object is automatically inferred.

    Examples
    ========

    >>> from sympy import Mul
    >>> from sympy.abc import x, y
    >>> Mul(x, 1)
    x
    >>> Mul(x, x)
    x**2

    If ``evaluate=False`` is passed, result is not evaluated.

    >>> Mul(1, 2, evaluate=False)
    1*2
    >>> Mul(x, x, evaluate=False)
    x*x

    ``Mul()`` also represents the general structure of multiplication
    operation.

    >>> from sympy import MatrixSymbol
    >>> A = MatrixSymbol('A', 2,2)
    >>> expr = Mul(x,y).subs({y:A})
    >>> expr
    x*A
    >>> type(expr)
    <class 'sympy.matrices.expressions.matmul.MatMul'>

    See Also
    ========

    MatMul

    """
    __slots__ = ()
    is_Mul = True
    _args_type = Expr
    _kind_dispatcher = KindDispatcher('Mul_kind_dispatcher', commutative=True)
    identity: ClassVar[Expr]
    if TYPE_CHECKING:

        def __new__(cls, *args: Expr | complex, evaluate: bool=True) -> Expr:
            ...

        @property
        def args(self) -> tuple[Expr, ...]:
            ...

    def __neg__(self):
        c, args = self.as_coeff_mul()
        if args[0] is not S.ComplexInfinity:
            c = -c
        if c is not S.One:
            if args[0].is_Number:
                args = list(args)
                if c is S.NegativeOne:
                    args[0] = -args[0]
                else:
                    args[0] *= c
            else:
                args = (c,) + args
        return self._from_args(args, self.is_commutative)

    @cacheit
    def as_coeff_mul(self, *deps, rational=True, **kwargs):
        if deps:
            l1, l2 = sift(self.args, lambda x: x.has(*deps), binary=True)
            return (self._new_rawargs(*l2), tuple(l1))
        args = self.args
        if args[0].is_Number:
            if not rational or args[0].is_Rational:
                return (args[0], args[1:])
            elif args[0].is_extended_negative:
                return (S.NegativeOne, (-args[0],) + args[1:])
        return (S.One, args)
    _eval_is_commutative = lambda self: _fuzzy_group((a.is_commutative for a in self.args))
