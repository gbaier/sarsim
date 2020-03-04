import numpy as np

def wrap_phase(phi):
    return ((phi + np.pi) % (2 * np.pi)) - np.pi
