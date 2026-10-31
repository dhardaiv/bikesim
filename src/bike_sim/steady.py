"""Steady-state speed and the best posture in a constant wind.

At equilibrium the rider holds critical power CP(u_a) and the net force is
zero. Scanning posture u_a and solving for v shows the posture-power
trade-off directly.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from . import rider
from .config import DEFAULT, Params
from .dynamics import forces


def equilibrium_speed(
    u_a: float,
    headwind: float,
    power: float | None = None,
    grade: float = 0.0,
    p: Params = DEFAULT,
) -> float:
    """Speed where net force is zero, at the given power (default CP(u_a))."""
    if power is None:
        power = rider.critical_power(u_a, p)
    wind = (-headwind, 0.0)

    def net(v: float) -> float:
        f_prop = p.eta * power / v
        _, f_aero, f_roll, f_grade = forces(v, power, u_a, wind, grade, p)
        return f_prop + f_aero + f_roll + f_grade

    return float(brentq(net, 0.05, 60.0, xtol=1e-9))


def optimal_posture(headwind: float, grade: float = 0.0, p: Params = DEFAULT) -> tuple[float, float]:
    """Return (u_a*, v*) maximizing steady speed at critical power."""
    res = minimize_scalar(
        lambda u: -equilibrium_speed(u, headwind, grade=grade, p=p),
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-5},
    )
    return float(res.x), -float(res.fun)


def tradeoff_table(headwinds: np.ndarray, p: Params = DEFAULT) -> dict[str, np.ndarray]:
    upright = np.array([equilibrium_speed(0.0, w, p=p) for w in headwinds])
    tuck = np.array([equilibrium_speed(1.0, w, p=p) for w in headwinds])
    opt = np.array([optimal_posture(w, p=p) for w in headwinds])
    return {
        "headwind": headwinds,
        "v_upright": upright,
        "v_tuck": tuck,
        "u_opt": opt[:, 0],
        "v_opt": opt[:, 1],
    }
