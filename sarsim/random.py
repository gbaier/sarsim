import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.linalg import toeplitz
from scipy.ndimage import convolve


def multivariate_complex_normal(cov, size=1):
    """ multivariate complex normal distribution

    :params cov: covariance matrix. Must be positive-definite
    :params size: number of samples
    """

    n_dim = cov.shape[0]

    # complex vector space isomorphism
    cov_real = np.kron(np.eye(2), cov.real) \
             + np.kron(np.array([[0, -1], [1, 0]]), cov.imag)

    # normalization
    cov_real /= 2

    xy = __multivariate_normal(cov_real, size)

    if xy.ndim == 1:
        return xy[:n_dim] + 1j * xy[-n_dim:]
    return xy[:, :n_dim] + 1j * xy[:, -n_dim:]


def __multivariate_normal(cov, size):
    # for some covariance matrices the cholesky decomposition may fail
    # in this case fall back to scipy's, i.e. numpy's routine, which relies
    # on singular value decomposition.
    # We try to use the Cholesky decomposition for performance reasons.
    # See https://github.com/numpy/numpy/pull/3938 for a dicussion.
    try:
        # Choleksy decomposition: COV = LL^H
        L = np.linalg.cholesky(cov)

        xy = norm.rvs(size=cov.shape[0] * size).reshape((-1, size))
        # so L @ xy has covariance LL^T = cov
        xy = np.dot(L, xy)

        # first dimension: number of samples:
        xy = xy.T

        if size == 1:
            xy = xy.flatten()

    except np.linalg.linalg.LinAlgError:
        xy = multivariate_normal.rvs(cov=cov, size=size)

    return xy


def multivariate_rayleigh(cov, size):
    """ multivariate rayleigh distribution

    :params cov: covariance matrix of the underlying Gaussian. Must be positive-definite
    :params size: number of samples

    """

    return np.abs(multivariate_complex_normal(cov, size))
