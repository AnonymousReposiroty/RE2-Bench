from numbers import Integral, Real
from sklearn.linear_model._base import (
    BaseEstimator,
    LinearClassifierMixin,
    SparseCoefMixin,
)
from sklearn.utils._param_validation import Hidden, Interval, StrOptions

class LogisticRegression(LinearClassifierMixin, SparseCoefMixin, BaseEstimator):
    """
    Logistic Regression (aka logit, MaxEnt) classifier.

    This class implements regularized logistic regression using a set of available
    solvers. **Note that regularization is applied by default**. It can handle both
    dense and sparse input `X`. Use C-ordered arrays or CSR matrices containing 64-bit
    floats for optimal performance; any other input format will be converted (and
    copied).

    The solvers 'lbfgs', 'newton-cg', 'newton-cholesky' and 'sag' support only L2
    regularization with primal formulation, or no regularization. The 'liblinear'
    solver supports both L1 and L2 regularization (but not both, i.e. elastic-net),
    with a dual formulation only for the L2 penalty. The Elastic-Net (combination of L1
    and L2) regularization is only supported by the 'saga' solver.

    For :term:`multiclass` problems (whenever `n_classes >= 3`), all solvers except
    'liblinear' optimize the (penalized) multinomial loss. 'liblinear' only handles
    binary classification but can be extended to handle multiclass by using
    :class:`~sklearn.multiclass.OneVsRestClassifier`.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    penalty : {'l1', 'l2', 'elasticnet', None}, default='l2'
        Specify the norm of the penalty:

        - `None`: no penalty is added;
        - `'l2'`: add a L2 penalty term and it is the default choice;
        - `'l1'`: add a L1 penalty term;
        - `'elasticnet'`: both L1 and L2 penalty terms are added.

        .. warning::
           Some penalties may not work with some solvers. See the parameter
           `solver` below, to know the compatibility between the penalty and
           solver.

        .. versionadded:: 0.19
           l1 penalty with SAGA solver (allowing 'multinomial' + L1)

        .. deprecated:: 1.8
           `penalty` was deprecated in version 1.8 and will be removed in 1.10.
           Use `l1_ratio` instead. `l1_ratio=0` for `penalty='l2'`, `l1_ratio=1` for
           `penalty='l1'` and `l1_ratio` set to any float between 0 and 1 for
           `'penalty='elasticnet'`.

    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization. `C=np.inf` results in unpenalized logistic regression.
        For a visual example on the effect of tuning the `C` parameter
        with an L1 penalty, see:
        :ref:`sphx_glr_auto_examples_linear_model_plot_logistic_path.py`.

    l1_ratio : float, default=0.0
        The Elastic-Net mixing parameter, with `0 <= l1_ratio <= 1`. Setting
        `l1_ratio=1` gives a pure L1-penalty, setting `l1_ratio=0` a pure L2-penalty.
        Any value between 0 and 1 gives an Elastic-Net penalty of the form
        `l1_ratio * L1 + (1 - l1_ratio) * L2`.

        .. warning::
           Certain values of `l1_ratio`, i.e. some penalties, may not work with some
           solvers. See the parameter `solver` below, to know the compatibility between
           the penalty and solver.

        .. versionchanged:: 1.8
            Default value changed from None to 0.0.

        .. deprecated:: 1.8
            `None` is deprecated and will be removed in version 1.10. Always use
            `l1_ratio` to specify the penalty type.

    dual : bool, default=False
        Dual (constrained) or primal (regularized, see also
        :ref:`this equation <regularized-logistic-loss>`) formulation. Dual formulation
        is only implemented for l2 penalty with liblinear solver. Prefer `dual=False`
        when n_samples > n_features.

    tol : float, default=1e-4
        Tolerance for stopping criteria.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    intercept_scaling : float, default=1
        Useful only when the solver `liblinear` is used
        and `self.fit_intercept` is set to `True`. In this case, `x` becomes
        `[x, self.intercept_scaling]`,
        i.e. a "synthetic" feature with constant value equal to
        `intercept_scaling` is appended to the instance vector.
        The intercept becomes
        ``intercept_scaling * synthetic_feature_weight``.

        .. note::
            The synthetic feature weight is subject to L1 or L2
            regularization as all other features.
            To lessen the effect of regularization on synthetic feature weight
            (and therefore on the intercept) `intercept_scaling` has to be increased.

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

        .. versionadded:: 0.17
           *class_weight='balanced'*

    random_state : int, RandomState instance, default=None
        Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
        data. See :term:`Glossary <random_state>` for details.

    solver : {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'},             default='lbfgs'

        Algorithm to use in the optimization problem. Default is 'lbfgs'.
        To choose a solver, you might want to consider the following aspects:

        - 'lbfgs' is a good default solver because it works reasonably well for a wide
          class of problems.
        - For :term:`multiclass` problems (`n_classes >= 3`), all solvers except
          'liblinear' minimize the full multinomial loss, 'liblinear' will raise an
          error.
        - 'newton-cholesky' is a good choice for
          `n_samples` >> `n_features * n_classes`, especially with one-hot encoded
          categorical features with rare categories. Be aware that the memory usage
          of this solver has a quadratic dependency on `n_features * n_classes`
          because it explicitly computes the full Hessian matrix.
        - For small datasets, 'liblinear' is a good choice, whereas 'sag'
          and 'saga' are faster for large ones;
        - 'liblinear' can only handle binary classification by default. To apply a
          one-versus-rest scheme for the multiclass setting one can wrap it with the
          :class:`~sklearn.multiclass.OneVsRestClassifier`.

        .. warning::
           The choice of the algorithm depends on the penalty chosen (`l1_ratio=0`
           for L2-penalty, `l1_ratio=1` for L1-penalty and `0 < l1_ratio < 1` for
           Elastic-Net) and on (multinomial) multiclass support:

           ================= ======================== ======================
           solver            l1_ratio                 multinomial multiclass
           ================= ======================== ======================
           'lbfgs'           l1_ratio=0               yes
           'liblinear'       l1_ratio=1 or l1_ratio=0 no
           'newton-cg'       l1_ratio=0               yes
           'newton-cholesky' l1_ratio=0               yes
           'sag'             l1_ratio=0               yes
           'saga'            0<=l1_ratio<=1           yes
           ================= ======================== ======================

        .. note::
           'sag' and 'saga' fast convergence is only guaranteed on features
           with approximately the same scale. You can preprocess the data with
           a scaler from :mod:`sklearn.preprocessing`.

        .. seealso::
           Refer to the :ref:`User Guide <Logistic_regression>` for more
           information regarding :class:`LogisticRegression` and more specifically the
           :ref:`Table <logistic_regression_solvers>`
           summarizing solver/penalty supports.

        .. versionadded:: 0.17
           Stochastic Average Gradient (SAG) descent solver. Multinomial support in
           version 0.18.
        .. versionadded:: 0.19
           SAGA solver.
        .. versionchanged:: 0.22
           The default solver changed from 'liblinear' to 'lbfgs' in 0.22.
        .. versionadded:: 1.2
           newton-cholesky solver. Multinomial support in version 1.6.

    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.

    verbose : int, default=0
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Useless for liblinear solver. See :term:`the Glossary <warm_start>`.

        .. versionadded:: 0.17
           *warm_start* to support *lbfgs*, *newton-cg*, *sag*, *saga* solvers.

    n_jobs : int, default=None
        Does not have any effect.

        .. deprecated:: 1.8
           `n_jobs` is deprecated in version 1.8 and will be removed in 1.10.

    Attributes
    ----------

    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.

        `coef_` is of shape (1, n_features) when the given problem is binary.

    intercept_ : ndarray of shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.

        If `fit_intercept` is set to False, the intercept is set to zero.
        `intercept_` is of shape (1,) when the given problem is binary.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : ndarray of shape (1, )
        Actual number of iterations for all classes.

        .. versionchanged:: 0.20

            In SciPy <= 1.0.0 the number of lbfgs iterations may exceed
            ``max_iter``. ``n_iter_`` will now report at most ``max_iter``.

    See Also
    --------
    SGDClassifier : Incrementally trained logistic regression (when given
        the parameter ``loss="log_loss"``).
    LogisticRegressionCV : Logistic regression with built-in cross validation.

    Notes
    -----
    The underlying C implementation uses a random number generator to
    select features when fitting the model. It is thus not uncommon,
    to have slightly different results for the same input data. If
    that happens, try with a smaller tol parameter.

    Predict output may not match that of standalone liblinear in certain
    cases. See :ref:`differences from liblinear <liblinear_differences>`
    in the narrative documentation.

    References
    ----------

    L-BFGS-B -- Software for Large-scale Bound-constrained Optimization
        Ciyou Zhu, Richard Byrd, Jorge Nocedal and Jose Luis Morales.
        http://users.iems.northwestern.edu/~nocedal/lbfgsb.html

    LIBLINEAR -- A Library for Large Linear Classification
        https://www.csie.ntu.edu.tw/~cjlin/liblinear/

    SAG -- Mark Schmidt, Nicolas Le Roux, and Francis Bach
        Minimizing Finite Sums with the Stochastic Average Gradient
        https://hal.inria.fr/hal-00860051/document

    SAGA -- Defazio, A., Bach F. & Lacoste-Julien S. (2014).
            :arxiv:`"SAGA: A Fast Incremental Gradient Method With Support
            for Non-Strongly Convex Composite Objectives" <1407.0202>`

    Hsiang-Fu Yu, Fang-Lan Huang, Chih-Jen Lin (2011). Dual coordinate descent
        methods for logistic regression and maximum entropy models.
        Machine Learning 85(1-2):41-75.
        https://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = LogisticRegression(random_state=0).fit(X, y)
    >>> clf.predict(X[:2, :])
    array([0, 0])
    >>> clf.predict_proba(X[:2, :])
    array([[9.82e-01, 1.82e-02, 1.44e-08],
           [9.72e-01, 2.82e-02, 3.02e-08]])
    >>> clf.score(X, y)
    0.97

    For a comparison of the LogisticRegression with other classifiers see:
    :ref:`sphx_glr_auto_examples_classification_plot_classification_probability.py`.
    """
    _parameter_constraints: dict = {'penalty': [StrOptions({'l1', 'l2', 'elasticnet'}), None, Hidden(StrOptions({'deprecated'}))], 'C': [Interval(Real, 0, None, closed='right')], 'l1_ratio': [Interval(Real, 0, 1, closed='both'), None], 'dual': ['boolean'], 'tol': [Interval(Real, 0, None, closed='left')], 'fit_intercept': ['boolean'], 'intercept_scaling': [Interval(Real, 0, None, closed='neither')], 'class_weight': [dict, StrOptions({'balanced'}), None], 'random_state': ['random_state'], 'solver': [StrOptions({'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'})], 'max_iter': [Interval(Integral, 0, None, closed='left')], 'verbose': ['verbose'], 'warm_start': ['boolean'], 'n_jobs': [None, Integral]}

    def __init__(self, penalty='deprecated', *, C=1.0, l1_ratio=0.0, dual=False, tol=0.0001, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, verbose=0, warm_start=False, n_jobs=None):
        self.penalty = penalty
        self.C = C
        self.l1_ratio = l1_ratio
        self.dual = dual
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        if self.solver == 'liblinear':
            tags.classifier_tags.multi_class = False
        return tags
