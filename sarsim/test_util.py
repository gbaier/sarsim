import numpy as np

from . import util


def test_wrap_phase():
    n_vals = 100
    wrapped_phase_vals = np.random.uniform(-np.pi, np.pi, n_vals)
    unwrapped_phase_vals = np.random.randint(
        0, 100, n_vals) * 2 * np.pi + wrapped_phase_vals

    np.testing.assert_array_almost_equal(
        util.wrap_phase(unwrapped_phase_vals), wrapped_phase_vals)
