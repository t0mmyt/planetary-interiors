from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.constants import G, pi

# Numpy optimised volume calculation
vol_layer = np.frompyfunc(
    lambda in_r, out_r: 4 / 3 * np.pi * (out_r**3 - in_r**3), 2, 1
)


def planet_core_pressure(mass: float, radius: float) -> float:
    # With corrective factor of 2 for planetary bodies
    return (2 * 3 * G * mass**2) / (8 * pi * radius**4)


class DensityModel(ABC):
    @abstractmethod
    def density_for_depth(self, depth: np.float64 | float) -> np.float64:
        pass

    @abstractmethod
    def mean_density(
        self,
    ) -> np.float64:
        pass


class SimpleDensityModel(DensityModel):
    def __init__(self, density: np.float64 | float):
        self._density = density

    @classmethod
    def with_known_mass(
        cls, volume: np.float64 | float, mass: np.float64 | float
    ) -> "SimpleDensityModel":
        return SimpleDensityModel(mass / volume)

    def density_for_depth(self, depth: np.float64 | float) -> np.float64:
        return self._density

    def mean_density(
        self,
    ) -> np.float64:
        return self._density


class OneDDensityModel(DensityModel):
    def __init__(
        self,
        depths: np.ndarray[np.float64],
        densities: np.ndarray[np.float64],
        upper_depth: np.float64 | float,
        lower_depth: np.float64 | float,
        total_radius: np.float64 | float,
    ):
        self.total_radius = total_radius
        self.lower_depth = lower_depth
        self.upper_depth = upper_depth
        if len(depths) != len(densities):
            raise ValueError("Depths and Densities were different lengths")
        self.depths = depths
        self.densities = densities

    @classmethod
    def from_csv(
        cls,
        path: str,
        upper_depth: np.float64 | float,
        lower_depth: np.float64 | float,
        total_radius: np.float64 | float,
    ):
        df = pd.read_csv(path)
        return OneDDensityModel(
            depths=df["depth_km"].to_numpy() * 1000,
            densities=df["density"].to_numpy(),
            upper_depth=upper_depth,
            lower_depth=lower_depth,
            total_radius=total_radius,
        )

    def density_for_depth(self, depth: np.float64 | float) -> np.float64:
        return np.float64(np.interp(depth, self.depths, self.densities))

    def mean_density(self) -> np.float64:
        inner_radius = np.float64(self.total_radius - self.lower_depth)
        outer_radius = np.float64(self.total_radius - self.upper_depth)
        depths = np.linspace(
            self.lower_depth, self.upper_depth, int(self.lower_depth - self.upper_depth)
        )
        # radii = np.linspace(inner_radius, outer_radius, int(self.lower_depth - self.upper_depth))

        densities = np.interp(depths, self.depths, self.densities)
        volumes = self._integrate_volumes(inner_radius, outer_radius)
        masses = densities * volumes
        mean_density = np.sum(masses) / np.sum(volumes)
        return np.float64(mean_density)

    @staticmethod
    def _integrate_volumes(
        inner_r: np.float64 | float, outer_r: np.float64 | float
    ) -> np.ndarray[np.float64]:
        """
        Returns an array of volumes of concentric spheres of 1m thickness

        Warning: total volume may be inaccurate for small thicknesses,
        suitable for use on the order of kilometres
        """
        radii = np.linspace(inner_r, outer_r, int(outer_r - inner_r))
        return vol_layer(radii - 0.5, radii + 0.5)
