from scipy.linalg import toeplitz
import numpy as np


def coh_thermal(signal_power, noise_power):
    """thermal coherence due to signal to noise ratio

    formula taken from: H. A. Zebker and J. Villasenor, "Decorrelation in interferometric radar
    echoes," in IEEE Transactions on Geoscience and Remote Sensing, vol. 30, no. 5, pp. 950-959,
    Sep 1992.

    """

    return signal_power/(signal_power+noise_power)


def exp_decay_coh_mat(M, lbda):
    """ generates a coherence matrix with exponential decay """

    coh_vec = np.exp(-np.arange(0, M) * lbda)

    return toeplitz(coh_vec)
