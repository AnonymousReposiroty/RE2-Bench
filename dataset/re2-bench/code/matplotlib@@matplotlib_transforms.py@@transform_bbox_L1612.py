import numpy as np

class Transform(TransformNode):
    """
    The base class of all `TransformNode` instances that
    actually perform a transformation.

    All non-affine transformations should be subclasses of this class.
    New affine transformations should be subclasses of `Affine2D`.

    Subclasses of this class should override the following members (at
    minimum):

    - :attr:`input_dims`
    - :attr:`output_dims`
    - :meth:`transform`
    - :meth:`inverted` (if an inverse exists)

    The following attributes may be overridden if the default is unsuitable:

    - :attr:`is_separable` (defaults to True for 1D -> 1D transforms, False
      otherwise)
    - :attr:`has_inverse` (defaults to True if :meth:`inverted` is overridden,
      False otherwise)

    If the transform needs to do something non-standard with
    `matplotlib.path.Path` objects, such as adding curves
    where there were once line segments, it should override:

    - :meth:`transform_path`
    """
    input_dims = None
    '\n    The number of input dimensions of this transform.\n    Must be overridden (with integers) in the subclass.\n    '
    output_dims = None
    '\n    The number of output dimensions of this transform.\n    Must be overridden (with integers) in the subclass.\n    '
    is_separable = False
    'True if this transform is separable in the x- and y- dimensions.'
    has_inverse = False
    'True if this transform has a corresponding inverse transform.'

    def transform(self, values):
        """
        Apply this transformation on the given array of *values*.

        Parameters
        ----------
        values : array-like
            The input values as an array of length :attr:`~Transform.input_dims` or
            shape (N, :attr:`~Transform.input_dims`).

        Returns
        -------
        array
            The output values as an array of length :attr:`~Transform.output_dims` or
            shape (N, :attr:`~Transform.output_dims`), depending on the input.
        """
        values = np.asanyarray(values)
        ndim = values.ndim
        values = values.reshape((-1, self.input_dims))
        res = self.transform_affine(self.transform_non_affine(values))
        if ndim == 0:
            assert not np.ma.is_masked(res)
            return res[0, 0]
        if ndim == 1:
            return res.reshape(-1)
        elif ndim == 2:
            return res
        raise ValueError('Input values must have shape (N, {dims}) or ({dims},)'.format(dims=self.input_dims))

    def transform_affine(self, values):
        """
        Apply only the affine part of this transformation on the
        given array of values.

        ``transform(values)`` is always equivalent to
        ``transform_affine(transform_non_affine(values))``.

        In non-affine transformations, this is generally a no-op.  In
        affine transformations, this is equivalent to
        ``transform(values)``.

        Parameters
        ----------
        values : array
            The input values as an array of length :attr:`~Transform.input_dims` or
            shape (N, :attr:`~Transform.input_dims`).

        Returns
        -------
        array
            The output values as an array of length :attr:`~Transform.output_dims` or
            shape (N, :attr:`~Transform.output_dims`), depending on the input.
        """
        return self.get_affine().transform(values)

    def transform_non_affine(self, values):
        """
        Apply only the non-affine part of this transformation.

        ``transform(values)`` is always equivalent to
        ``transform_affine(transform_non_affine(values))``.

        In non-affine transformations, this is generally equivalent to
        ``transform(values)``.  In affine transformations, this is
        always a no-op.

        Parameters
        ----------
        values : array
            The input values as an array of length
            :attr:`~matplotlib.transforms.Transform.input_dims` or
            shape (N, :attr:`~matplotlib.transforms.Transform.input_dims`).

        Returns
        -------
        array
            The output values as an array of length
            :attr:`~matplotlib.transforms.Transform.output_dims` or shape
            (N, :attr:`~matplotlib.transforms.Transform.output_dims`),
            depending on the input.
        """
        return values

    def transform_bbox(self, bbox):
        """
        Transform the given bounding box.

        For smarter transforms including caching (a common requirement in
        Matplotlib), see `TransformedBbox`.
        """
        return Bbox(self.transform(bbox.get_points()))

    def get_affine(self):
        """Get the affine part of this transform."""
        return IdentityTransform()
