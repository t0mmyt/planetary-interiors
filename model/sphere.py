from dataclasses import dataclass

import numpy as np
from matplotlib import patheffects
from matplotlib.axes import Axes

from model.density import DensityModel

# Numpy optimised volume calculation
vol_layer = np.frompyfunc(
    lambda in_r, out_r: 4 / 3 * np.pi * (out_r**3 - in_r**3), 2, 1
)


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
    name: str
    inner: Sphere
    outer: Sphere
    color: str = None
    density_model: DensityModel = None
    _density: np.float64 = None
    _mass: np.float64 = None

    @classmethod
    def of_km(cls, name: str, inner_r, outer_r, color: str = None) -> "Layer":
        return Layer(
            name=name,
            inner=Sphere.of_km(inner_r),
            outer=Sphere.of_km(outer_r),
            color=color,
        )

    @property
    def volume(self) -> np.float64:
        return self.outer.volume - self.inner.volume

    @property
    def thickness(self) -> np.float64:
        return self.outer.radius - self.inner.radius

    def set_mass(self, mass: np.float64 | float) -> "Layer":
        self._mass = np.float64(mass)
        self._density = self._mass / self.volume
        return self

    @property
    def mass(self) -> np.float64:
        return self.density_model.mean_density() * self.volume

    def set_density_model(self, dm: DensityModel) -> "Layer":
        self.density_model = dm
        return self

    @property
    def density(self) -> np.float64:
        if self.density_model:
            return self.density_model.mean_density()
        return self._density

    def density_for_depth(self, meters: np.float64 | float) -> np.float64:
        if self.density_model:
            return self.density_model.density_for_depth(meters)
        return self._density


@dataclass
class Planet:
    total_mass: np.float64
    layers: [Layer]

    @property
    def radius(self):
        return np.max([layer.outer.radius for layer in self.layers])

    @property
    def calculated_mass(self):
        return np.sum([layer.mass for layer in self.layers])

    def density_for_depth(self, meters: np.float64 | float) -> np.float64:
        layer = list(
            filter(
                lambda x: x.inner.radius <= self.radius - meters <= x.outer.radius,
                self.layers,
            )
        )
        if len(layer) != 1:
            raise ValueError(f"Density not found for depth {meters}m")
        return layer[0].density_for_depth(meters)


def plot_layers(planet: Planet, ax: Axes):
    r = planet.radius
    for layer in planet.layers:
        inner_r = layer.inner.radius
        outer_r = layer.outer.radius
        ax.fill(
            *layer_poly(inner_r=inner_r, outer_r=outer_r, total_r=r),
            color=layer.color,
            alpha=0.8,
        )
        text = ax.text(
            x=0,
            y=r / 1000 - np.mean((inner_r / 1000, outer_r / 1000)),
            s=f"{layer.name} $\\bar{{\\rho}}={layer.density / 1000:.2f}$",
            ha="right",
            va="center",
        )
        text.set_path_effects([patheffects.withStroke(linewidth=3, foreground="w")])
    ax.set_aspect(1)
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylabel("depth (km)")
    ax.set_xlabel("radius (km)")
    ax.set_title("Layers")


def layer_poly(inner_r, outer_r, total_r):
    # Convert to km for plot
    inner_r /= 1000
    outer_r /= 1000
    total_r /= 1000
    arc_points = np.linspace(90, 180, 90)
    inner_x = np.cos(np.deg2rad(arc_points)) * inner_r * -1
    inner_y = total_r - np.sin(np.deg2rad(arc_points)) * inner_r
    outer_x = np.cos(np.deg2rad(np.flip(arc_points))) * outer_r * -1
    outer_y = total_r - np.sin(np.deg2rad(np.flip(arc_points))) * outer_r
    x = np.concatenate((inner_x, outer_x))
    y = np.concatenate((inner_y, outer_y))
    return x, y


def plot_density_function(planet: Planet, ax: Axes, linspace=1000):
    y = np.linspace(0, planet.radius, linspace) / 1000
    x = np.array(list(map(planet.density_for_depth, y * 1000))) / 1000
    ax.plot(x, y)
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xlabel("$\\rho$ $(\\mathrm{g.cm}^{-3})$")
    ax.set_title("Density Profile")
    ax.set_yticks(
        np.array(list(planet.radius - layer.inner.radius for layer in planet.layers))
        / 1000
    )
