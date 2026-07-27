from sympy.matrices.dense import eye
from sympy.core.numbers import Rational, oo
from sympy.matrices.dense import Matrix
from sympy.matrices.dense import Matrix

class LRASolver:
    """
    Linear Arithmetic Solver for DPLL(T) implemented with an algorithm based on
    the Dual Simplex method. Uses Bland's pivoting rule to avoid cycling.

    References
    ==========

    .. [1] Dutertre, B., de Moura, L.:
           A Fast Linear-Arithmetic Solver for DPLL(T)
           https://link.springer.com/chapter/10.1007/11817963_11
    """

    def __init__(self, A, slack_variables, nonslack_variables, enc_to_boundary, s_subs, testing_mode):
        """
        Use the "from_encoded_cnf" method to create a new LRASolver.
        """
        self.run_checks = testing_mode
        self.s_subs = s_subs
        if any((not isinstance(a, Rational) for a in A)):
            raise UnhandledInput('Non-rational numbers are not handled')
        if any((not isinstance(b.bound, Rational) for b in enc_to_boundary.values())):
            raise UnhandledInput('Non-rational numbers are not handled')
        m, n = (len(slack_variables), len(slack_variables) + len(nonslack_variables))
        if m != 0:
            assert A.shape == (m, n)
        if self.run_checks:
            assert A[:, n - m:] == -eye(m)
        self.enc_to_boundary = enc_to_boundary
        self.boundary_to_enc = {value: key for key, value in enc_to_boundary.items()}
        self.A = A
        self.slack = slack_variables
        self.nonslack = nonslack_variables
        self.all_var = nonslack_variables + slack_variables
        self.slack_set = set(slack_variables)
        self.is_sat = True
        self.result = None

    def check(self):
        """
        Searches for an assignment that satisfies all constraints
        or determines that no such assignment exists and gives
        a minimal conflict clause that "explains" why the
        constraints are unsatisfiable.

        Returns
        =======

        (True, assignment) or (False, explanation)

        assignment : dict of LRAVariables to values
            Assigned values are tuples that represent a rational number
            plus some infinatesimal delta.

        explanation : set of ints
        """
        if self.is_sat:
            return (True, {var: var.assign for var in self.all_var})
        if self.result:
            return self.result
        from sympy.matrices.dense import Matrix
        M = self.A.copy()
        basic = {s: i for i, s in enumerate(self.slack)}
        nonbasic = set(self.nonslack)
        while True:
            if self.run_checks:
                assert all(((nb.assign >= nb.lower) == True and (nb.assign <= nb.upper) == True for nb in nonbasic))
                if all((v.assign[0] != float('inf') and v.assign[0] != -float('inf') for v in self.all_var)):
                    X = Matrix([v.assign[0] for v in self.all_var])
                    assert all((abs(val) < 10 ** (-10) for val in M * X))
                assert all((x.upper[1] <= 0 for x in self.all_var))
                assert all((x.lower[1] >= 0 for x in self.all_var))
            cand = [b for b in basic if b.assign < b.lower or b.assign > b.upper]
            if len(cand) == 0:
                return (True, {var: var.assign for var in self.all_var})
            xi = min(cand, key=lambda v: v.col_idx)
            i = basic[xi]
            if xi.assign < xi.lower:
                cand = [nb for nb in nonbasic if M[i, nb.col_idx] > 0 and nb.assign < nb.upper or (M[i, nb.col_idx] < 0 and nb.assign > nb.lower)]
                if len(cand) == 0:
                    N_plus = [nb for nb in nonbasic if M[i, nb.col_idx] > 0]
                    N_minus = [nb for nb in nonbasic if M[i, nb.col_idx] < 0]
                    conflict = []
                    conflict += [Boundary.from_upper(nb) for nb in N_plus]
                    conflict += [Boundary.from_lower(nb) for nb in N_minus]
                    conflict.append(Boundary.from_lower(xi))
                    conflict = [-neg * self.boundary_to_enc[c] for c, neg in conflict]
                    return (False, conflict)
                xj = min(cand, key=str)
                M = self._pivot_and_update(M, basic, nonbasic, xi, xj, xi.lower)
            if xi.assign > xi.upper:
                cand = [nb for nb in nonbasic if M[i, nb.col_idx] < 0 and nb.assign < nb.upper or (M[i, nb.col_idx] > 0 and nb.assign > nb.lower)]
                if len(cand) == 0:
                    N_plus = [nb for nb in nonbasic if M[i, nb.col_idx] > 0]
                    N_minus = [nb for nb in nonbasic if M[i, nb.col_idx] < 0]
                    conflict = []
                    conflict += [Boundary.from_upper(nb) for nb in N_minus]
                    conflict += [Boundary.from_lower(nb) for nb in N_plus]
                    conflict.append(Boundary.from_upper(xi))
                    conflict = [-neg * self.boundary_to_enc[c] for c, neg in conflict]
                    return (False, conflict)
                xj = min(cand, key=lambda v: v.col_idx)
                M = self._pivot_and_update(M, basic, nonbasic, xi, xj, xi.upper)

    def _pivot_and_update(self, M, basic, nonbasic, xi, xj, v):
        """
        Pivots basic variable xi with nonbasic variable xj,
        and sets value of xi to v and adjusts the values of all basic variables
        to keep equations satisfied.
        """
        i, j = (basic[xi], xj.col_idx)
        assert M[i, j] != 0
        theta = (v - xi.assign) * (1 / M[i, j])
        xi.assign = v
        xj.assign = xj.assign + theta
        for xk in basic:
            if xk != xi:
                k = basic[xk]
                akj = M[k, j]
                xk.assign = xk.assign + theta * akj
        basic[xj] = basic[xi]
        del basic[xi]
        nonbasic.add(xi)
        nonbasic.remove(xj)
        return self._pivot(M, i, j)

    @staticmethod
    def _pivot(M, i, j):
        """
        Performs a pivot operation about entry i, j of M by performing
        a series of row operations on a copy of M and returning the result.
        The original M is left unmodified.

        Conceptually, M represents a system of equations and pivoting
        can be thought of as rearranging equation i to be in terms of
        variable j and then substituting in the rest of the equations
        to get rid of other occurances of variable j.

        Example
        =======

        >>> from sympy.matrices.dense import Matrix
        >>> from sympy.logic.algorithms.lra_theory import LRASolver
        >>> from sympy import var
        >>> Matrix(3, 3, var('a:i'))
        Matrix([
        [a, b, c],
        [d, e, f],
        [g, h, i]])

        This matrix is equivalent to:
        0 = a*x + b*y + c*z
        0 = d*x + e*y + f*z
        0 = g*x + h*y + i*z

        >>> LRASolver._pivot(_, 1, 0)
        Matrix([
        [ 0, -a*e/d + b, -a*f/d + c],
        [-1,       -e/d,       -f/d],
        [ 0,  h - e*g/d,  i - f*g/d]])

        We rearrange equation 1 in terms of variable 0 (x)
        and substitute to remove x from the other equations.

        0 = 0 + (-a*e/d + b)*y + (-a*f/d + c)*z
        0 = -x + (-e/d)*y + (-f/d)*z
        0 = 0 + (h - e*g/d)*y + (i - f*g/d)*z
        """
        _, _, Mij = (M[i, :], M[:, j], M[i, j])
        if Mij == 0:
            raise ZeroDivisionError('Tried to pivot about zero-valued entry.')
        A = M.copy()
        A[i, :] = -A[i, :] / Mij
        for row in range(M.shape[0]):
            if row != i:
                A[row, :] = A[row, :] + A[row, j] * A[i, :]
        return A
