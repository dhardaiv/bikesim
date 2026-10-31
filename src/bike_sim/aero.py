"""Quasi-steady aerodynamics of a cyclist.

Quasi-steady assumption: the drag at each instant equals the steady drag at
the instantaneous relative air velocity. This holds while the wind changes
slowly compared with the flow-through time L / V (about 0.1 s for a rider).
Gusts with timescales near that, or vortex-shedding dynamics, would require
an unsteady model or a time-resolved CFD solve.
"""

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT, Params


@dataclass(frozen=True)
class AeroState:
    v_rel: float  # along-track air speed relative to rider (positive = air coming at rider)
    airspeed: float  # magnitude of relative air velocity
    yaw: float  # rad, angle of relative wind from the bike's axis
    q: float  # Pa, dynamic pressure
    reynolds: float
    cda: float  # effective CdA after yaw and Re corrections
    drag_x: float  # N, along-track aero force (negative = resisting)
    pressure_drag: float  # N, magnitude
    friction_drag: float  # N, magnitude


def dynamic_pressure(airspeed: float, p: Params = DEFAULT) -> float:
    return 0.5 * p.rho * airspeed**2


def reynolds(airspeed: float, p: Params = DEFAULT) -> float:
    return p.rho * airspeed * p.char_length / p.mu


def effective_cda(cda: float, yaw: float, re: float, p: Params = DEFAULT) -> float:
    yaw_factor = 1.0 + p.k_yaw * np.sin(yaw) ** 2
    re_factor = 1.0
    if p.cd_re_slope != 0.0 and re > 0.0:
        re_factor = 1.0 + p.cd_re_slope * np.log10(re / p.re_ref)
    return cda * yaw_factor * re_factor


def aero(v: float, wind_x: float, wind_y: float, cda: float, p: Params = DEFAULT) -> AeroState:
    """Aero force on a rider moving at speed v along +x in wind (wind_x, wind_y).

    wind_x < 0 is a headwind. The along-track force is
    F_x = -0.5 * rho * CdA_eff * |V| * v_rel, which is resistive for v_rel > 0
    and propulsive when a tailwind outruns the rider.
    """
    v_rel = v - wind_x
    cross = -wind_y
    airspeed = float(np.hypot(v_rel, cross))
    yaw = float(np.arctan2(abs(cross), abs(v_rel))) if airspeed > 0 else 0.0
    re = reynolds(airspeed, p)
    cda_eff = effective_cda(cda, yaw, re, p)
    q = dynamic_pressure(airspeed, p)
    drag_x = -0.5 * p.rho * cda_eff * airspeed * v_rel
    total = abs(q * cda_eff)
    return AeroState(
        v_rel=v_rel,
        airspeed=airspeed,
        yaw=yaw,
        q=q,
        reynolds=re,
        cda=cda_eff,
        drag_x=drag_x,
        pressure_drag=(1.0 - p.skin_friction_fraction) * total,
        friction_drag=p.skin_friction_fraction * total,
    )
