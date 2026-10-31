"""Spatial wind fields w(x) along the course. Negative wind_x is a headwind."""

from dataclasses import dataclass

import numpy as np


class WindField:
    def __call__(self, x: float) -> tuple[float, float]:
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantWind(WindField):
    headwind: float = 0.0
    crosswind: float = 0.0

    def __call__(self, x: float) -> tuple[float, float]:
        return -self.headwind, self.crosswind


@dataclass(frozen=True)
class SinusoidalGusts(WindField):
    mean_headwind: float = 2.0
    amplitude: float = 4.0
    wavelength: float = 800.0  # m
    crosswind: float = 0.0

    def __call__(self, x: float) -> tuple[float, float]:
        head = self.mean_headwind + self.amplitude * np.sin(2.0 * np.pi * x / self.wavelength)
        return -float(head), self.crosswind


@dataclass(frozen=True)
class HeadwindPatch(WindField):
    """Mild background wind with a strong headwind section, smoothed at the edges."""

    background_headwind: float = 1.0
    patch_headwind: float = 7.0
    start: float = 2000.0
    end: float = 3000.0
    edge: float = 50.0  # m, transition width
    crosswind: float = 0.0

    def __call__(self, x: float) -> tuple[float, float]:
        rise = 0.5 * (1.0 + np.tanh((x - self.start) / self.edge))
        fall = 0.5 * (1.0 - np.tanh((x - self.end) / self.edge))
        head = self.background_headwind + (self.patch_headwind - self.background_headwind) * rise * fall
        return -float(head), self.crosswind
