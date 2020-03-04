from . import backscatter

import numpy as np

def test_incidence_angle_flat():
    """ test incidence angle computation for flat terrain """

    terrain = np.zeros((5, 5))
    look_angle = 30

    inc_angle, layover, shadowing = backscatter.incidence_angle(terrain, look_angle)

    np.testing.assert_almost_equal(inc_angle, np.radians(look_angle)*np.ones_like(terrain))
    np.testing.assert_array_equal(shadowing, np.zeros(terrain.shape, dtype=np.bool))

def test_incidence_angle_pos_slope():
    terrain = np.arange(10).reshape(2,5)
    look_angle = 20

    inc_angle, layover, shadowing = backscatter.incidence_angle(terrain, look_angle)

    np.testing.assert_almost_equal(inc_angle, -np.radians(25)*np.ones_like(terrain))


def test_incidence_angle_neg_slope():
    terrain = np.arange(10, 0, -1).reshape(2,5)
    look_angle = 20

    inc_angle, layover, shadowing = backscatter.incidence_angle(terrain, look_angle)

    np.testing.assert_almost_equal(inc_angle, np.radians(65)*np.ones_like(terrain))


def test_incidence_angle_sanity():
    terrain = np.random.normal(size=(20, 20))

    look_angle_1 = 20
    look_angle_2 = 30

    inc_angle_1, layover_1, shadowing_1 = backscatter.incidence_angle(terrain, look_angle_1)
    inc_angle_2, layover_2, shadowing_2 = backscatter.incidence_angle(terrain, look_angle_2)

    # less shadow for smaller look angle
    assert np.sum(shadowing_1) <= np.sum(shadowing_2)

    # less layover for larger look angle
    assert np.sum(layover_2) <= np.sum(layover_1)

    np.testing.assert_array_less(inc_angle_1[~shadowing_2], inc_angle_2[~shadowing_2])
