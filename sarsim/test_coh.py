import numpy as np

from . import coh

def test_exp_decay_coh_mat():
    M = 7

    cmat = coh.exp_decay_coh_mat(M, 0.1)

    assert cmat.shape == (M, M)
    np.testing.assert_equal(cmat, cmat.T)
    np.testing.assert_equal(cmat[0], np.sort(cmat[0])[::-1])
    np.testing.assert_equal(np.diag(cmat), np.ones(M))
