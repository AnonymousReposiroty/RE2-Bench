import numpy as np

class _ReducedHCT_Element:
    """
    Implementation of reduced HCT triangular element with explicit shape
    functions.

    Computes z, dz, d2z and the element stiffness matrix for bending energy:
    E(f) = integral( (d2z/dx2 + d2z/dy2)**2 dA)

    *** Reference for the shape functions: ***
    [1] Basis functions for general Hsieh-Clough-Tocher _triangles, complete or
        reduced.
        Michel Bernadou, Kamal Hassan
        International Journal for Numerical Methods in Engineering.
        17(5):784 - 789.  2.01

    *** Element description: ***
    9 dofs: z and dz given at 3 apex
    C1 (conform)

    """
    M = np.array([[0.0, 0.0, 0.0, 4.5, 4.5, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.25, 0.0, 0.0, 0.5, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.25, 0.0, 0.0, 1.25, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 1.0, 0.0, -1.5, 0.0, 3.0, 3.0, 0.0, 0.0, 3.0], [0.0, 0.0, 0.0, -0.25, 0.25, 0.0, 1.0, 0.0, 0.0, 0.5], [0.25, 0.0, 0.0, -0.5, -0.25, 1.0, 0.0, 0.0, 0.0, 1.0], [0.5, 0.0, 1.0, 0.0, -1.5, 0.0, 0.0, 3.0, 3.0, 3.0], [0.25, 0.0, 0.0, -0.25, -0.5, 0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.25, -0.25, 0.0, 0.0, 1.0, 0.0, 0.5]])
    M0 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 1.5, 1.5, 0.0, 0.0, 0.0, 0.0, -3.0], [-0.5, 0.0, 0.0, 0.75, 0.75, 0.0, 0.0, 0.0, 0.0, -1.5], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, -1.5, -1.5, 0.0, 0.0, 0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, -0.75, -0.75, 0.0, 0.0, 0.0, 0.0, 1.5]])
    M1 = np.array([[-0.5, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.25, 0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.25, 0.0, 0.0, -0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    M2 = np.array([[0.5, 0.0, 0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0], [0.25, 0.0, 0.0, 0.0, -0.75, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.25, 0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    rotate_dV = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [-1.0, -1.0], [-1.0, -1.0], [1.0, 0.0]])
    rotate_d2V = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, -2.0, -1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [-2.0, 0.0, -1.0]])
    n_gauss = 9
    gauss_pts = np.array([[13.0 / 18.0, 4.0 / 18.0, 1.0 / 18.0], [4.0 / 18.0, 13.0 / 18.0, 1.0 / 18.0], [7.0 / 18.0, 7.0 / 18.0, 4.0 / 18.0], [1.0 / 18.0, 13.0 / 18.0, 4.0 / 18.0], [1.0 / 18.0, 4.0 / 18.0, 13.0 / 18.0], [4.0 / 18.0, 7.0 / 18.0, 7.0 / 18.0], [4.0 / 18.0, 1.0 / 18.0, 13.0 / 18.0], [13.0 / 18.0, 1.0 / 18.0, 4.0 / 18.0], [7.0 / 18.0, 4.0 / 18.0, 7.0 / 18.0]], dtype=np.float64)
    gauss_w = np.ones([9], dtype=np.float64) / 9.0
    E = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    J0_to_J1 = np.array([[-1.0, 1.0], [-1.0, 0.0]])
    J0_to_J2 = np.array([[0.0, -1.0], [1.0, -1.0]])

    def get_function_derivatives(self, alpha, J, ecc, dofs):
        """
        Parameters
        ----------
        *alpha* is a (N x 3 x 1) array (array of column-matrices of
        barycentric coordinates)
        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
        triangle first apex)
        *ecc* is a (N x 3 x 1) array (array of column-matrices of triangle
        eccentricities)
        *dofs* is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
        degrees of freedom.

        Returns
        -------
        Returns the values of interpolated function derivatives [dz/dx, dz/dy]
        in global coordinates at locations alpha, as a column-matrices of
        shape (N x 2 x 1).
        """
        subtri = np.argmin(alpha, axis=1)[:, 0]
        ksi = _roll_vectorized(alpha, -subtri, axis=0)
        E = _roll_vectorized(ecc, -subtri, axis=0)
        x = ksi[:, 0, 0]
        y = ksi[:, 1, 0]
        z = ksi[:, 2, 0]
        x_sq = x * x
        y_sq = y * y
        z_sq = z * z
        dV = _to_matrix_vectorized([[-3.0 * x_sq, -3.0 * x_sq], [3.0 * y_sq, 0.0], [0.0, 3.0 * z_sq], [-2.0 * x * z, -2.0 * x * z + x_sq], [-2.0 * x * y + x_sq, -2.0 * x * y], [2.0 * x * y - y_sq, -y_sq], [2.0 * y * z, y_sq], [z_sq, 2.0 * y * z], [-z_sq, 2.0 * x * z - z_sq], [x * z - y * z, x * y - y * z]])
        dV = dV @ _extract_submatrices(self.rotate_dV, subtri, block_size=2, axis=0)
        prod = self.M @ dV
        prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ dV)
        prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ dV)
        prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ dV)
        dsdksi = _roll_vectorized(prod, 3 * subtri, axis=0)
        dfdksi = dofs @ dsdksi
        J_inv = _safe_inv22_vectorized(J)
        dfdx = J_inv @ _transpose_vectorized(dfdksi)
        return dfdx
