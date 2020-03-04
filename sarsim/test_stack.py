from itertools import islice

import numpy as np
import pytest

from . import stack


def test_gen_noisy_stack():
    stack_dims = (3, 4, 5)
    amps = np.ones(stack_dims)
    phis = np.zeros(stack_dims)
    cohs = 0.7*np.ones((stack_dims[0], *stack_dims))

    assert stack_dims == stack.gen_noisy_stack(amps, phis, cohs).shape


def test_amp_phi_coh3cov_exceptions():

    # test for amp and phi vectors
    with pytest.raises(ValueError):
        amp = np.ones((3, 2))
        coh = np.ones((amp.size, amp.size))
        phi = np.ones(4)
        stack.amp_phi_coh2cov(amp, phi, coh)

    # test for matching sizes
    with pytest.raises(ValueError):
        amp = np.ones(3)
        coh = np.ones((amp.size, amp.size))
        phi = np.ones(4)
        stack.amp_phi_coh2cov(amp, phi, coh)

    # test for quadratic coherence
    with pytest.raises(ValueError):
        amp = np.ones(3)
        coh = np.ones((3, 4))
        phi = np.ones(4)
        stack.amp_phi_coh2cov(amp, phi, coh)


def test_amp_phi_coh2cov():
    n_dim = 3
    amp = np.ones(n_dim)
    phi = np.ones(n_dim)
    coh = np.eye(n_dim)

    cov = stack.amp_phi_coh2cov(amp, phi, coh)

    assert cov.shape == (n_dim, n_dim)
    assert cov.dtype == np.complex
    np.testing.assert_equal(cov, cov.T.conj())


def test_amps_phis_cohs2covs():

    shape = (7, 5, 3)

    amps = np.ones(shape)
    phis = np.ones(shape)
    cohs = np.ones((shape[0], *shape))

    covs = stack.amps_phis_cohs2covs(amps, phis, cohs)

    assert covs.shape == cohs.shape

    covs_swap = np.swapaxes(covs, 0, 1)
    np.testing.assert_equal(covs, covs_swap)


def test_too_close():

    outl = (3, 4, 5)
    min_dis = (0, 2, 2)
    prev_outls = [
        (0, 0, 0),
        (3, 8, 9),
        (3, 5, 9),
    ]

    assert stack.too_close(outl, prev_outls, min_dis)
    assert not stack.too_close(outl, prev_outls[:-1], min_dis)


def test_gen_outliers():
    amp = 1
    n_outliers = 1000

    outliers = islice(stack.gen_outliers(amp), n_outliers)

    # No possible unit test I could think of. Just checking that calling
    # stack.gen_outliers does not throw any exceptions
    assert True


def test_gen_coordinates():
    stack_shape = (5, 4, 2)
    n_outliers = 1000

    coords = islice(stack.gen_coordinates(stack_shape), n_outliers)

    assert all(c >= (0, 0, 0) for c in coords)
    assert all(c < stack_shape for c in coords)


def test_add_outliers2arr():
    stk = np.zeros((1, 4, 5), dtype=np.complex)
    coords = [(0, 1, 2), (0, 3, 0)]
    outliers = [1j, 1]
    selem = np.ones((1, 3, 3))

    des_stk = np.array([[[0, 1j, 1j, 1j, 0],
                         [0, 1j, 1j, 1j, 0],
                         [1, 1+1j, 1j, 1j, 0],
                         [1, 1, 0, 0, 0]]])

    np.testing.assert_array_equal(stack.add_outliers2arr(stk, zip(outliers, coords), selem), des_stk)
