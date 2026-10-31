"""Longitudinal rider dynamics: m dv/dt = F_prop + F_aero + F_roll + F_grade."""

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from . import rider
from .aero import aero
from .config import DEFAULT, Params
from .wind import WindField


@dataclass(frozen=True)
class RiderState:
    x: float = 0.0
    v: float = 0.0
    w_bal: float = DEFAULT.w_prime_max
    posture: float = 0.0
    t: float = 0.0


def flat(x: float) -> float:
    return 0.0


def forces(
    v: float,
    power: float,
    posture: float,
    wind: tuple[float, float],
    grade: float = 0.0,
    p: Params = DEFAULT,
) -> tuple[float, float, float, float]:
    """Return (F_prop, F_aero, F_roll, F_grade) in newtons."""
    f_prop = min(p.eta * power / max(v, p.v_eps), p.f_prop_max)
    f_aero = aero(v, wind[0], wind[1], rider.cda(posture, p), p).drag_x
    theta = np.arctan(grade)
    f_roll = -p.crr * p.mass * p.g * np.cos(theta) if v > 0.0 else 0.0
    f_grade = -p.mass * p.g * np.sin(theta)
    return f_prop, f_aero, f_roll, f_grade


def acceleration(v, power, posture, wind, grade=0.0, p: Params = DEFAULT) -> float:
    return sum(forces(v, power, posture, wind, grade, p)) / p.mass


def substep(
    s: RiderState,
    u_p: float,
    posture_target: float,
    wind_field: WindField,
    grade_fn: Callable[[float], float] = flat,
    p: Params = DEFAULT,
) -> tuple[RiderState, float]:
    """Advance one physics substep (semi-implicit Euler). Returns (state, power)."""
    dt = p.dt_sim
    max_dp = p.posture_rate * dt
    posture = s.posture + float(np.clip(posture_target - s.posture, -max_dp, max_dp))
    power = rider.deliverable_power(u_p, posture, s.w_bal, p)
    a = acceleration(s.v, power, posture, wind_field(s.x), grade_fn(s.x), p)
    v = max(s.v + a * dt, 0.0)
    x = s.x + v * dt
    w_bal = rider.update_w_bal(s.w_bal, power, posture, dt, p)
    return replace(s, x=x, v=v, w_bal=w_bal, posture=posture, t=s.t + dt), power
