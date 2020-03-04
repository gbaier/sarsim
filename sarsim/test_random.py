import numpy as np

from . import random

def test_multivariate_complex_normal_shape_and_type():
    n_dim = 3
    n_samples = 100
    cov = np.eye(n_dim)
    samples = random.multivariate_complex_normal(cov, n_samples)

    assert samples.shape == (n_samples, n_dim)
    assert samples.dtype == np.complex


def test_multivariate_complex_normal_covariance():
    n_dim = 3
    n_samples = 10**6

    cov = np.exp(1j*np.random.uniform(-np.pi, np.pi, size=(n_dim, n_dim)))
    cov = np.dot(cov, cov.T.conj())

    samples = random.multivariate_complex_normal(cov, n_samples)

    # sample covariance matrix
    scm = np.cov(samples.T)

    np.testing.assert_almost_equal(scm, cov, decimal=2)
