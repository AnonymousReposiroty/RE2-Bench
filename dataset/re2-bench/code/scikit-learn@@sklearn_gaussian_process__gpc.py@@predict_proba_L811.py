from numbers import Integral
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context, clone
from sklearn.gaussian_process.kernels import RBF, CompoundKernel, Kernel
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

class GaussianProcessClassifier(ClassifierMixin, BaseEstimator):
    """Gaussian process classification (GPC) based on Laplace approximation.

    The implementation is based on Algorithm 3.1, 3.2, and 5.1 from [RW2006]_.

    Internally, the Laplace approximation is used for approximating the
    non-Gaussian posterior by a Gaussian.

    Currently, the implementation is restricted to using the logistic link
    function. For multi-class classification, several binary one-versus rest
    classifiers are fitted. Note that this class thus does not implement
    a true multi-class Laplace approximation.

    Read more in the :ref:`User Guide <gaussian_process>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    kernel : kernel instance, default=None
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are optimized during fitting. Also kernel
        cannot be a `CompoundKernel`.

    optimizer : 'fmin_l_bfgs_b', callable or None, default='fmin_l_bfgs_b'
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the  signature::

            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' is the objective function to be maximized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min

        Per default, the 'L-BFGS-B' algorithm from scipy.optimize.minimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::

            'fmin_l_bfgs_b'

    n_restarts_optimizer : int, default=0
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer=0 implies that one
        run is performed.

    max_iter_predict : int, default=100
        The maximum number of iterations in Newton's method for approximating
        the posterior during predict. Smaller values will reduce computation
        time at the cost of worse results.

    warm_start : bool, default=False
        If warm-starts are enabled, the solution of the last Newton iteration
        on the Laplace approximation of the posterior mode is used as
        initialization for the next call of _posterior_mode(). This can speed
        up convergence when _posterior_mode is called several times on similar
        problems as in hyperparameter optimization. See :term:`the Glossary
        <warm_start>`.

    copy_X_train : bool, default=True
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    multi_class : {'one_vs_rest', 'one_vs_one'}, default='one_vs_rest'
        Specifies how multi-class classification problems are handled.
        Supported are 'one_vs_rest' and 'one_vs_one'. In 'one_vs_rest',
        one binary Gaussian process classifier is fitted for each class, which
        is trained to separate this class from the rest. In 'one_vs_one', one
        binary Gaussian process classifier is fitted for each pair of classes,
        which is trained to separate these two classes. The predictions of
        these binary predictors are combined into multi-class predictions.
        Note that 'one_vs_one' does not support predicting probability
        estimates.

    n_jobs : int, default=None
        The number of jobs to use for the computation: the specified
        multiclass problems are computed in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    base_estimator_ : ``Estimator`` instance
        The estimator instance that defines the likelihood function
        using the observed data.

    kernel_ : kernel instance
        The kernel used for prediction. In case of binary classification,
        the structure of the kernel is the same as the one passed as parameter
        but with optimized hyperparameters. In case of multi-class
        classification, a CompoundKernel is returned which consists of the
        different kernels used in the one-versus-rest classifiers.

    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``

    classes_ : array-like of shape (n_classes,)
        Unique class labels.

    n_classes_ : int
        The number of classes in the training data

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    GaussianProcessRegressor : Gaussian process regression (GPR).

    References
    ----------
    .. [RW2006] `Carl E. Rasmussen and Christopher K.I. Williams,
       "Gaussian Processes for Machine Learning",
       MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import RBF
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = 1.0 * RBF(1.0)
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9866...
    >>> gpc.predict_proba(X[:2,:])
    array([[0.83548752, 0.03228706, 0.13222543],
           [0.79064206, 0.06525643, 0.14410151]])

    For a comparison of the GaussianProcessClassifier with other classifiers see:
    :ref:`sphx_glr_auto_examples_classification_plot_classification_probability.py`.
    """
    _parameter_constraints: dict = {'kernel': [Kernel, None], 'optimizer': [StrOptions({'fmin_l_bfgs_b'}), callable, None], 'n_restarts_optimizer': [Interval(Integral, 0, None, closed='left')], 'max_iter_predict': [Interval(Integral, 1, None, closed='left')], 'warm_start': ['boolean'], 'copy_X_train': ['boolean'], 'random_state': ['random_state'], 'multi_class': [StrOptions({'one_vs_rest', 'one_vs_one'})], 'n_jobs': [Integral, None]}

    def __init__(self, kernel=None, *, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=None):
        self.kernel = kernel
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.warm_start = warm_start
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        self.multi_class = multi_class
        self.n_jobs = n_jobs

    def predict_proba(self, X):
        """Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated for classification.

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        if self.n_classes_ > 2 and self.multi_class == 'one_vs_one':
            raise ValueError('one_vs_one multi-class mode does not support predicting probability estimates. Use one_vs_rest mode instead.')
        if self.kernel is None or self.kernel.requires_vector_input:
            X = validate_data(self, X, ensure_2d=True, dtype='numeric', reset=False)
        else:
            X = validate_data(self, X, ensure_2d=False, dtype=None, reset=False)
        return self.base_estimator_.predict_proba(X)
