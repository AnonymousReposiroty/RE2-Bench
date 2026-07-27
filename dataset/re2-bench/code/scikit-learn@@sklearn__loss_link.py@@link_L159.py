import numpy as np

class IdentityLink(BaseLink):
    """The identity link function g(x)=x."""

    def link(self, y_pred, out=None):
        if out is not None:
            np.copyto(out, y_pred)
            return out
        else:
            return y_pred
    inverse = link
