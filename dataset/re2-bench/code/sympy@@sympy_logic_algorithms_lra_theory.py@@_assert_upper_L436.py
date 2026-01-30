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

    def _assert_upper(self, xi, ci, from_equality=False, from_neg=False):
        """
        Adjusts the upper bound on variable xi if the new upper bound is
        more limiting. The assignment of variable xi is adjusted to be
        within the new bound if needed.

        Also calls `self._update` to update the assignment for slack variables
        to keep all equalities satisfied.
        """
        if self.result:
            assert self.result[0] != False
        self.result = None
        if ci >= xi.upper:
            return None
        if ci < xi.lower:
            assert (xi.lower[1] >= 0) is True
            assert (ci[1] <= 0) is True
            lit1, neg1 = Boundary.from_lower(xi)
            lit2 = Boundary(var=xi, const=ci[0], strict=ci[1] != 0, upper=True, equality=from_equality)
            if from_neg:
                lit2 = lit2.get_negated()
            neg2 = -1 if from_neg else 1
            conflict = [-neg1 * self.boundary_to_enc[lit1], -neg2 * self.boundary_to_enc[lit2]]
            self.result = (False, conflict)
            return self.result
        xi.upper = ci
        xi.upper_from_eq = from_equality
        xi.upper_from_neg = from_neg
        if xi in self.nonslack and xi.assign > ci:
            self._update(xi, ci)
        if self.run_checks and all((v.assign[0] != float('inf') and v.assign[0] != -float('inf') for v in self.all_var)):
            M = self.A
            X = Matrix([v.assign[0] for v in self.all_var])
            assert all((abs(val) < 10 ** (-10) for val in M * X))
        return None

    def _update(self, xi, v):
        """
        Updates all slack variables that have equations that contain
        variable xi so that they stay satisfied given xi is equal to v.
        """
        i = xi.col_idx
        for j, b in enumerate(self.slack):
            aji = self.A[j, i]
            b.assign = b.assign + (v - xi.assign) * aji
        xi.assign = v
