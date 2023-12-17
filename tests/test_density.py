import numpy as np
import pytest

from model.density import OneDDensityModel


class TestDensityModel:
    @pytest.mark.parametrize(
        "inner_r,outer_r,volume",
        [
            (0, 50e3, 5.23599e14),
            (0, 100e3, 4.18879e15),
            (50e3, 100e3, 3.665191e15),
            (0, 3390e3, 1.631878e20),
        ],
    )
    def test__integrate_volumes(self, inner_r, outer_r, volume):
        calc_vol = np.sum(OneDDensityModel._integrate_volumes(inner_r, outer_r))
        assert np.isclose(calc_vol, volume)
