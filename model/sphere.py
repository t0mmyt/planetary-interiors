from dataclasses import dataclass

import numpy as np


@dataclass
class Sphere:
    radius: np.float64

    @property
    def volume(self) -> np.float64:
        return 4 / 3 * np.pi * self.radius**3

    @property
    def circumference(self) -> np.float64:
        return np.float64(2 * np.pi * self.radius)

    @classmethod
    def of_km(cls, km: float | np.float64) -> "Sphere":
        return Sphere(radius=np.float64(km) * 1000)


@dataclass
class Layer:
    inner: Sphere
    outer: Sphere

    @classmethod
    def of_km(cls, inner_r, outer_r) -> "Layer":
        return Layer(inner=Sphere.of_km(inner_r), outer=Sphere.of_km(outer_r))

    @property
    def volume(self) -> np.float64:
        return self.outer.volume - self.inner.volume

    @property
    def thickness(self) -> np.float64:
        return self.outer.radius - self.inner.radius


@dataclass
class Planet:
    layers: [Layer]
