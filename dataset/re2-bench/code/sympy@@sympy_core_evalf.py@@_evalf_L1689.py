from sympy.core.expr import Expr

class EvalfMixin:
    """Mixin class adding evalf capability."""
    __slots__: tuple[str, ...] = ()
    n = evalf

    def _evalf(self, prec: int) -> Expr:
        """Helper for evalf. Does the same thing but takes binary precision"""
        r = self._eval_evalf(prec)
        if r is None:
            r = self
        return r

    def _eval_evalf(self, prec: int) -> Expr | None:
        return None
