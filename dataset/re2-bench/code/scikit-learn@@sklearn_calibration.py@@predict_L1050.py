from scipy.special import expit
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    RegressorMixin,
    _fit_context,
    clone,
)
from sklearn.utils import Bunch, _safe_indexing, column_or_1d, get_tags, indexable

class _SigmoidCalibration(RegressorMixin, BaseEstimator):
    """Sigmoid regression model.

    Attributes
    ----------
    a_ : float
        The slope.

    b_ : float
        The intercept.
    """

    def predict(self, T):
        """Predict new data by linear interpolation.

        Parameters
        ----------
        T : array-like of shape (n_samples,)
            Data to predict from.

        Returns
        -------
        T_ : ndarray of shape (n_samples,)
            The predicted data.
        """
        T = column_or_1d(T)
        return expit(-(self.a_ * T + self.b_))
