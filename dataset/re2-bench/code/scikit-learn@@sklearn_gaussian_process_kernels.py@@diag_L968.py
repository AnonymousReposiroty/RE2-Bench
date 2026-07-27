class Product(KernelOperator):
    """The `Product` kernel takes two kernels :math:`k_1` and :math:`k_2`
    and combines them via

    .. math::
        k_{prod}(X, Y) = k_1(X, Y) * k_2(X, Y)

    Note that the `__mul__` magic method is overridden, so
    `Product(RBF(), RBF())` is equivalent to using the * operator
    with `RBF() * RBF()`.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    k1 : Kernel
        The first base-kernel of the product-kernel

    k2 : Kernel
        The second base-kernel of the product-kernel


    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import (RBF, Product,
    ...            ConstantKernel)
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = Product(ConstantKernel(2), RBF())
    >>> gpr = GaussianProcessRegressor(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    1.0
    >>> kernel
    1.41**2 * RBF(length_scale=1)
    """

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return self.k1.diag(X) * self.k2.diag(X)
