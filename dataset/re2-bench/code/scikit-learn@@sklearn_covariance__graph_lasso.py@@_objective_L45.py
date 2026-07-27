import numpy as np
from sklearn.covariance import EmpiricalCovariance, empirical_covariance, log_likelihood

def _objective(mle, precision_, alpha):
    """Evaluation of the graphical-lasso objective function

    the objective function is made of a shifted scaled version of the
    normalized log-likelihood (i.e. its empirical mean over the samples) and a
    penalisation term to promote sparsity
    """
    p = precision_.shape[0]
    cost = -2.0 * log_likelihood(mle, precision_) + p * np.log(2 * np.pi)
    cost += alpha * (np.abs(precision_).sum() - np.abs(np.diag(precision_)).sum())
    return cost
