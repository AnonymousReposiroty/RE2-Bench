import numpy as np

class _DOF_estimator_geom(_DOF_estimator):
    """Fast 'geometric' approximation, recommended for large arrays."""

    def compute_geom_grads(self):
        """
        Compute the (global) gradient component of f assumed linear (~f).
        returns array df of shape (nelems, 2)
        df[ielem].dM[ielem] = dz[ielem] i.e. df = dz x dM = dM.T^-1 x dz
        """
        tris_pts = self._tris_pts
        tris_f = self.z[self._triangles]
        dM1 = tris_pts[:, 1, :] - tris_pts[:, 0, :]
        dM2 = tris_pts[:, 2, :] - tris_pts[:, 0, :]
        dM = np.dstack([dM1, dM2])
        dM_inv = _safe_inv22_vectorized(dM)
        dZ1 = tris_f[:, 1] - tris_f[:, 0]
        dZ2 = tris_f[:, 2] - tris_f[:, 0]
        dZ = np.vstack([dZ1, dZ2]).T
        df = np.empty_like(dZ)
        df[:, 0] = dZ[:, 0] * dM_inv[:, 0, 0] + dZ[:, 1] * dM_inv[:, 1, 0]
        df[:, 1] = dZ[:, 0] * dM_inv[:, 0, 1] + dZ[:, 1] * dM_inv[:, 1, 1]
        return df
