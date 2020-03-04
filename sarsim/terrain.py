import numpy as np
from itertools import product
from scipy.ndimage.filters import gaussian_filter


def diamond_square(size=33, ampl=1, smoothing=1, seed=None):
    assert np.log2(size-1).is_integer()

    terrain = np.zeros((size, size))
    max_offset = size-1
    offsets = [2**x for x in range(int(np.log2(max_offset)))][::-1]

    np.random.seed(seed)
    terrain = init_corners(terrain)

    for of in offsets:
        diamond_step(terrain, of, ampl)
        square_step(terrain, of, ampl)
    terrain -= terrain.min()
    terrain /= terrain.max()
    return gaussian_filter(terrain, sigma=smoothing)


def step(terrain, offset, ampl, coords, get_neighbours):
    scale_factor = ampl*offset/terrain.shape[0]
    for coord in coords:
        neighbours = get_neighbours(coord, offset)
        n = 0
        for nb in neighbours:
            try:
                terrain[coord] += terrain[nb]
                n += 1
            except IndexError:
                pass
        terrain[coord] /= n
        terrain[coord] += scale_factor*np.random.uniform(-1, 1)
    return terrain


def diamond_step(terrain, offset, ampl):
    def get_neighbours(coord, offset):
        ngbs = [
                (coord[0]-offset, coord[1]-offset),
                (coord[0]-offset, coord[1]+offset),
                (coord[0]+offset, coord[1]-offset),
                (coord[0]+offset, coord[1]+offset),
                ]
        return ngbs
    coords = list(range(offset, terrain.shape[0], 2*offset))
    coords = product(coords, coords)
    return step(terrain, offset, ampl, coords, get_neighbours)


def square_step(terrain, offset, ampl):
    def get_neighbours(coord, offset):
        ngbs = [
                (coord[0]-offset, coord[1]),
                (coord[0]+offset, coord[1]),
                (coord[0], coord[1]-offset),
                (coord[0], coord[1]+offset),
                ]
        return ngbs
    c1 = list(range(offset, terrain.shape[0], 2*offset))
    c2 = list(range(0, terrain.shape[0], 2*offset))
    coords = list(product(c1, c2)) + list(product(c2, c1))
    return step(terrain, offset, ampl, coords, get_neighbours)


def init_corners(nparr):
    nparr[0, 0] = np.random.exponential()
    nparr[0, -1] = np.random.exponential()
    nparr[-1, 0] = np.random.exponential()
    nparr[-1, -1] = np.random.exponential()
    return nparr
