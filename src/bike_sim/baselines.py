"""Hand-designed policies. Each maps an env (reading its true state) to an action."""

from functools import lru_cache

import numpy as np

from . import rider
from .dynamics import acceleration
from .env import BikeWindEnv
from .steady import optimal_posture


def _hold_cp(env: BikeWindEnv, u_a: float) -> np.ndarray:
    u_p = rider.critical_power(u_a, env.p) / rider.p_cap(u_a, env.p)
    return env.encode_action(u_p, u_a)


def upright_cp(env: BikeWindEnv) -> np.ndarray:
    return _hold_cp(env, 0.0)


def tuck_cp(env: BikeWindEnv) -> np.ndarray:
    return _hold_cp(env, 1.0)


@lru_cache(maxsize=512)
def _steady_posture(headwind_rounded: float) -> float:
    return optimal_posture(headwind_rounded)[0]


def steady_optimal(env: BikeWindEnv) -> np.ndarray:
    """Posture that is steady-state optimal for the local wind, at critical power."""
    headwind = -env.wind_field(env.state.x)[0]
    return _hold_cp(env, _steady_posture(round(headwind, 1)))


_GRID = np.linspace(0.0, 1.0, 21)


def myopic(env: BikeWindEnv) -> np.ndarray:
    """Greedy: pick (u_a, u_p) maximizing instantaneous dv/dt, ignoring W' and the future."""
    s = env.state
    wind = env.wind_field(s.x)
    grade = env.grade_fn(s.x)
    best, best_acc = (1.0, 0.0), -np.inf
    for u_a in _GRID:
        for u_p in _GRID:
            power = rider.deliverable_power(u_p, u_a, s.w_bal, env.p)
            acc = acceleration(s.v, power, u_a, wind, grade, env.p)
            if acc > best_acc:
                best, best_acc = (u_p, u_a), acc
    return env.encode_action(*best)


BASELINES = {
    "upright_cp": upright_cp,
    "tuck_cp": tuck_cp,
    "steady_optimal": steady_optimal,
    "myopic": myopic,
}
