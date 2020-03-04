import numpy as np
from scipy.ndimage import convolve
from scipy.stats import rice
from itertools import islice

from . import random


def gen_noisy_stack(amps, phis, cohs):
    """ generates a stack of noisy and correlated SLCs

    :params amps: 3D numpy array of the SLCs amplitude
    :params phis: 3D numpy array of the SLCs phases
    :params cohs: 4D numpy array of all SLCs coherence combinations

    """

    slcs = np.empty(amps.shape, dtype=np.complex)

    covs = amps_phis_cohs2covs(amps, phis, cohs)

    for x in range(slcs.shape[1]):
        for y in range(slcs.shape[2]):
            slcs[:, x, y] = random.multivariate_complex_normal(
                covs[:, :, x, y])

    return slcs


def gen_noisy_slcs(amp, phi, coh):
    def noisegen(shape):
        real = np.random.standard_normal(shape)
        imag = np.random.standard_normal(shape)
        return (real + 1j * imag) / np.sqrt(2)

    noise1 = noisegen(amp.shape)
    noise2 = noisegen(amp.shape)

    l2 = amp * coh * np.exp(-1j * phi)
    l3 = amp * np.sqrt(1 - coh**2)

    slc1 = amp * noise1
    slc2 = l2 * noise1 + l3 * noise2

    return (slc1, slc2)


def amp_phi_coh2cov(amp, phi, coh):
    """ creates a covariance matrix from given amplidute and phase vectors.
    Coherence is a quadratic matrix with matching dimensions.

    """

    if amp.ndim != 1 or phi.ndim != 1:
        raise ValueError('amp and phi must be one dimensional')
    if coh.ndim != 2 or coh.shape[0] != coh.shape[1]:
        raise ValueError(
            'coh must be two dimensional and quadratic. Shape is {}'.format(
                coh.shape))
    if amp.size != phi.size or coh.shape[0] != amp.size:
        raise ValueError('dimension mismatch')

    slc = amp * np.exp(1j * phi)

    return coh * np.outer(slc, slc.conj())


def amps_phis_cohs2covs(amps, phis, cohs):
    """ creates covariance matrices for a stack of SAR SLCs defined by amps, phis and cohs """

    if amps.ndim != 3 or phis.ndim != 3:
        raise ValueError('amp and phi must be three dimensional')
    if cohs.ndim != 4 or cohs.shape[0] != cohs.shape[1]:
        raise ValueError(
            'coh must be four dimensional and first two dimensions must have equal size. Shape is {}'
            .format(cohs.shape))
    if amps.shape != phis.shape or cohs.shape[1:] != amps.shape:
        raise ValueError('dimension mismatch')

    covs = np.empty(cohs.shape, dtype=np.complex)

    for x in range(amps.shape[1]):
        for y in range(amps.shape[2]):
            covs[:, :, x, y] = amp_phi_coh2cov(amps[:, x, y], phis[:, x, y],
                                               cohs[:, :, x, y])

    return covs


def too_close(new_outl, prev_outls, min_dis):
    """ returns true if any of the distances between

    new_outl to prev_outls is smaller than min_dis

    """

    for po in prev_outls:
        dis = (abs(x - y) for x, y in zip(new_outl, po))
        if all(x < y for x, y in zip(dis, min_dis)):
            return True
    return False


def gen_outliers(amp):
    """ an infinite generator of outliers

    Outliers have Ricean distributed amplitude and uniformly distributed phase between -pi and pi


    :param amp: float
        amplitude of the Rice line-of-sight component

    :returns: the outliers

    """

    return iter(
        lambda: rice.rvs(amp) * np.exp(1j * np.random.uniform(-np.pi, np.pi)), 1)


def gen_coordinates(stack_shape, min_dis=None):
    """ an infinite generator of coordinates with a minimum distance between them

    :params stack_shape: tuple, shape of the stack,
        i.e. interval for the outliers
    """

    # no restrictions
    if min_dis is None:
        min_dis = (0, ) * len(stack_shape)

    coords = []
    for coord in iter(
            lambda: tuple(np.random.randint(0, x) for x in stack_shape), 1):
        if not too_close(coord, coords, min_dis):
            coords.append(coord)
            yield coord


def get_n_outlier_pairs(stack_shape, amp, n_outliers, min_dis=None):
    return islice(
        zip(gen_outliers(amp)), gen_coordinates(stack_shape, min_dis),
        n_outliers)


def add_outliers2arr(arr, outlier_pairs, selem=np.ones((1, 3, 3))):
    outlier_arr = np.zeros_like(arr)
    for outl, coord in outlier_pairs:
        outlier_arr[coord] = outl

    if outlier_arr.dtype == np.complex:
        outlier_arr_conv = convolve(
            outlier_arr.real, selem, mode='constant') + 1j * convolve(
                outlier_arr.imag, selem, mode='constant')
    else:
        outlier_arr_conv = convolve(outlier_arr, selem, mode='constant')

    mask = outlier_arr_conv != 0
    arr[mask] = outlier_arr_conv[mask]

    return arr


def get_outliers(shape, n_outliers):
    outliers = rice.rvs(
        4, size=n_outliers) * np.exp(
            1j * np.random.uniform(-np.pi, np.pi, size=n_outliers))
    x_coords = np.random.randint(0, shape[0], size=n_outliers)
    y_coords = np.random.randint(0, shape[1], size=n_outliers)
    for outlier, x, y in zip(outliers, x_coords, y_coords):
        yield outlier, (x, y)


def add_outliers(array, n_outliers):
    for outlier, (x, y) in get_outliers(array.shape, n_outliers):
        array[x, y] = outlier
    return array
