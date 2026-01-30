from .matexpr import MatrixExpr
from .special import Identity
from sympy.core import S
from sympy.matrices import MatrixBase
from .inverse import Inverse
from sympy.matrices.expressions import Inverse

class MatPow(MatrixExpr):

    def doit(self, **hints):
        if hints.get('deep', True):
            base, exp = (arg.doit(**hints) for arg in self.args)
        else:
            base, exp = self.args
        while isinstance(base, MatPow):
            exp *= base.args[1]
            base = base.args[0]
        if isinstance(base, MatrixBase):
            return base ** exp
        if exp == S.One:
            return base
        if exp == S.Zero:
            return Identity(base.rows)
        if exp == S.NegativeOne:
            from sympy.matrices.expressions import Inverse
            return Inverse(base).doit(**hints)
        eval_power = getattr(base, '_eval_power', None)
        if eval_power is not None:
            return eval_power(exp)
        return MatPow(base, exp)
