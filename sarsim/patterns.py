""" returns different patterns in the range 0 to 1"""

import numpy as np


def plateau(shape, mid_width=20):
    """
    generates a step image of size shape
    with a pedestal in the middle of size mid_width x mid_width
    """

    def midslice(dim, mid_width):
        start = dim//2 - mid_width//2
        return slice(start, start + mid_width)

    xslice = midslice(shape[0], mid_width)
    yslice = midslice(shape[1], mid_width)

    data = np.zeros(shape)
    data[xslice, yslice] = 1

    return data


def const_slope(shape):

    data = np.arange(shape[1]) / shape[1]
    data = np.tile(data, (shape[0], 1))

    return data


def sine(shape, omega=0.2):

    data = 0.5*(np.sin(np.arange(shape[1]) * omega) + 1)
    data = np.tile(data, (shape[0], 1))

    return data

def logsin(shape, doublerate=50):
    data = 0.5*(np.sin(2*np.pi*np.exp(np.log(2)*np.arange(shape[1])/doublerate))+1)
    data = np.tile(data, (shape[0], 1))

    return data

def logfreq(shape, doublerate=50):
    data =  np.angle(np.exp(1j*2*np.pi*np.exp(np.log(2)*np.arange(shape[1])/doublerate)))
    return np.tile(data, (shape[0], 1))

def logbar(shape, doublerate=50):
    data = logsin(shape, doublerate)
    data[data > 0.5] = 1
    data[data <= 0.5] = 0

    return data


def step_slope(shape):
    data = const_slope(shape)

    nstart = shape[1]//4
    nstop = nstart + shape[1]//2

    # auto broadcast array shapes
    data[:, :] -= data[0, nstart]
    data[:, :nstart] = 0

    data[:, nstop:] = data[0, nstop]

    data /= data.max()

    return data


def unit_step(shape):
    mid = shape[1]//2

    data = np.zeros(shape)

    data[:, mid:] = 1.0

    return data


def raised_cos(shape, beta=0.5):
    """
    returns a phase image with profile following the
    Fourier transform in the frequency domain
    """

    T = 2*np.pi

    f = np.linspace(0, 1/T, shape[1])
    H = np.zeros_like(f)

    idxs = f < (1+beta)/(2*T)
    H[idxs] = (1 + np.cos(np.pi*T/beta * (f[idxs] - (1-beta)/(2*T))))/2

    H[f < (1-beta)/(2*T)] = 1
    H = np.tile(H, (shape[0], 1))

    return H


def peaks(shape):
    xs = np.linspace(-3, 3, shape[1])
    ys = np.linspace(-3, 3, shape[0])

    x, y = np.meshgrid(xs, ys)

    data = 3*(1-x)**2*np.exp(-x**2 - (y+1)**2) \
        - 10*(x/5 - x**3 - y**5)*np.exp(-x**2-y**2) \
        - 1/3*np.exp(-(x+1)**2 - y**2)
    data -= data.min()
    data /= data.max()
    return data


def banana(shape, a=1, b=100):
    """
    Rosenbrock's banana function
    f(x,y) = (a-x)^2 + b*(y-x^2)^2
    """

    xs = np.linspace(-2, 2, shape[1])
    ys = np.linspace(-1.75, 2.25, shape[0])

    xv, yv = np.meshgrid(xs, ys)
    data = (a-xv)**2 + b*(yv-xv**2)**2

    data /= data.max()

    return data


def zebra(shape, period=20):

    data = np.zeros(shape)
    hp = period // 2
    for i in range(0, shape[1], period):
        data[:, i:i+hp] = 1

    return data


def squares(shape, period=20):
    data1 = zebra(shape, period)
    data2 = zebra((shape[1], shape[0]))

    return data1*data2.T != 0


def checkers(shape, period=20):
    data = np.zeros(shape)
    hp = period // 2
    s1, s2 = 0, hp
    for x in range(0, shape[0], hp):
        for y in range(s1, shape[1], period):
            data[x:x+hp, y:y+hp] = 1
        s1, s2 = s2, s1

    return data

def pizza(shape, nslices=12):
    assert nslices % 2 == 0
    xs = np.linspace(-1.0, 1.0, shape[0])
    ys = np.linspace(-1.0, 1.0, shape[1])
    real, imag = np.meshgrid(xs, ys)
    z = real+1j*imag
    data = np.zeros(z.shape)
    data[abs(z) < 1] = 1
    angles = np.linspace(-np.pi, np.pi, nslices+1)
    a1s = angles[::2]
    a2s = angles[1::2]
    for a1, a2 in zip(a1s, a2s):
       m1 = a1 < np.angle(z)
       m2 = np.angle(z) < a2
       m = np.logical_and(m1, m2)
       data[m] = 0
    return data
