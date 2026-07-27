from typing import TYPE_CHECKING, ClassVar, overload, Literal
from .basic import Basic, _args_sortkey
from .operations import AssocOp, AssocOpDispatcher
from .logic import fuzzy_not, _fuzzy_group
from .expr import Expr
from .kind import KindDispatcher
from .add import Add, _unevaluated_Add
from sympy.simplify.radsimp import fraction
from sympy.simplify.radsimp import fraction
from sympy.simplify.radsimp import fraction
from sympy.simplify.radsimp import fraction

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

    def _eval_expand_mul(self, **hints):
        from sympy.simplify.radsimp import fraction
        expr = self
        n, d = fraction(expr, hints.get('exact', False))
        if d.is_Mul:
            n, d = [i._eval_expand_mul(**hints) if i.is_Mul else i for i in (n, d)]
        expr = n / d
        if not expr.is_Mul:
            return expr
        plain, sums, rewrite = ([], [], False)
        for factor in expr.args:
            if factor.is_Add:
                sums.append(factor)
                rewrite = True
            elif factor.is_commutative:
                plain.append(factor)
            else:
                sums.append(Basic(factor))
        if not rewrite:
            return expr
        else:
            plain = self.func(*plain)
            if sums:
                deep = hints.get('deep', False)
                terms = self.func._expandsums(sums)
                args = []
                for term in terms:
                    t = self.func(plain, term)
                    if t.is_Mul and any((a.is_Add for a in t.args)) and deep:
                        t = t._eval_expand_mul()
                    args.append(t)
                return Add(*args)
            else:
                return plain
    _eval_is_commutative = lambda self: _fuzzy_group((a.is_commutative for a in self.args))
