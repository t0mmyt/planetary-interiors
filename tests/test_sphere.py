import numpy as np
import pytest

from model.sphere import Sphere, Planet, Layer, vol_layer


@pytest.mark.parametrize(
    "inner_r,outer_r,volume",
    [
        (0, 1, 4 / 3 * np.pi),
        (0, 500, 5.23599e8),
        (0, 1e3, 4.18879e9),
        (500, 1000, 4.18879e9 - 5.23599e8),
        (0, 3390e3, 1.631878e20),
    ],
)
def test_vol_sphere(inner_r, outer_r, volume):
    assert np.isclose(vol_layer(inner_r, outer_r), volume)


@pytest.mark.parametrize(
    "radius_km,radius_m,volume_m3,circumference_m",
    [(1.0, 1000.0, 4.18879e9, 6.28319e3), (0, 0, 0, 0)],
)
def test_volume_circumference(radius_km, radius_m, volume_m3, circumference_m):
    s = Sphere.of_km(km=np.float64(radius_km))
    assert np.isclose(s.radius, radius_m)
    assert np.isclose(s.volume, volume_m3)
    assert np.isclose(s.circumference, circumference_m)


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (((0, 50), (50, 100)), 4.18879e15),
    ],
)
def test_planet(inputs, expected):
    p = Planet(layers=[Layer.of_km(*r) for r in inputs], total_mass=np.float64(1.0))
    vol_sum = np.sum([layer.volume for layer in p.layers])
    assert np.isclose(vol_sum, expected)
